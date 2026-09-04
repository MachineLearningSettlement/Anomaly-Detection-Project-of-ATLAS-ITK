"""
Llama 3.2 Vision — DPO post-training for pixel-sheet retrieval.


------------------------------------------------------------------------------
THE MAIN AIM OF THE POST-TRAINING: 
     We want to post-train Llama 3.2 Vision so that to enrich our train data and eval data with efficient and 
     equilibrated samples for more robust and generalized learning. So given a description such as “find pixel 
     sheets with this geometry” or “find images containing dust anomalies on such specific pixel-sheet
     geometry,” we are going to train on an equilibrated normal dataset accounting for a wider variety of shapes, and also 
     for a plausible evaluation, wehere can use a variety of anomalous images presented on a variety of shapes, so 
     we could be able to assess the performance on a large scale.
------------------------------------------------------------------------------     

BLOCK 1 — MODEL PREPARATION
    Pretrained Llama 3.2 Vision → LoRA target modules → LoRA adapters

BLOCK 2 — DATA PREPARATION
    Existing preference dataset → Prompt + chosen/rejected images
    → one DPO pair per chosen/rejected pair → multimodal tokenization

BLOCK 3 — DPO TRAINING
    Policy/reference forward passes → chosen/rejected log-probabilities
    → DPO loss → backpropagation → update LoRA parameters

BLOCK 4 — ADAPTED MODEL
    Updated LoRA + frozen Llama backbone → adapted model
    → evaluation on unseen prompts → representative retrieval
    for PatchCore and AnomalyDINO.

DATASET
    Llama3.2_Vision_DPO_PixelSheet_Retrieval

    1. NORMAL GEOMETRY RETRIEVAL
       Prompt describing pixel-sheet shape/geometry + matching normal
       chosen image + different normal rejected images.
       The normal-image pool contains the overall normal-image set,
       including real normal images and synthetic images obtained from SD3.

    2. ANOMALY + GEOMETRY RETRIEVAL
       Prompt describing pixel-sheet shape/geometry and anomaly type
       + matching anomalous chosen image + different anomalous rejected images.

    NUM_REJECTED_PER_CHOSEN controls the number of rejected images paired
    with each chosen image.

IMPORTANT IMPLEMENTATION DETAIL
    Llama 3.2 Vision is a vision-language model whose output is text.
    Therefore, the technically valid DPO formulation presents BOTH candidate
    images in the same multimodal prompt and trains the model to prefer the
    candidate corresponding to the chosen image ("Image 1" or "Image 2").
    This preserves the intended image-preference objective while using the
    standard DPO preference loss over textual responses.
"""

import copy
import random
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig


# ===========================================================
# CONFIGURATION
# ===========================================================

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

DATASET_NAME = "Llama3.2_Vision_DPO_PixelSheet_Retrieval"
DATASET_PATH = "llama32_vision_dpo_pixel_sheet_retrieval"

# Hyperparameter requested for rejected-image selection.
NUM_REJECTED_PER_CHOSEN = 30

BETA = 0.1
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 5e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_LENGTH = 2048

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Q/K/V/O attention projections and FFN projections.
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

OUTPUT_DIR = "./llama32_vision_pixel_sheet_dpo_adapter"
USE_4BIT = True
SEED = 42

DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)


# ============================================================
# UTILITIES
# ============================================================

def load_rgb(path):
    return Image.open(str(path)).convert("RGB")


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_messages(prompt, image_a, image_b):
    """Same multimodal prompt for both DPO alternatives."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt + "\n\nImage 1:"},
                {"type": "image"},
                {"type": "text", "text": "\n\nImage 2:"},
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "\n\nWhich image best matches the description? "
                        "Answer only with Image 1 or Image 2."
                    ),
                },
            ],
        }
    ]


def make_chat_text(processor, messages):
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def tokenize_example(processor, prompt, chosen_image, rejected_image):
    # Randomize candidate order so the model cannot learn that Image 1 is
    # always preferred merely because it appears first.
    chosen_first = random.random() < 0.5

    if chosen_first:
        image_a, image_b = chosen_image, rejected_image
        chosen_answer, rejected_answer = "Image 1", "Image 2"
    else:
        image_a, image_b = rejected_image, chosen_image
        chosen_answer, rejected_answer = "Image 2", "Image 1"

    messages = build_messages(prompt, image_a, image_b)
    text = make_chat_text(processor, messages)

    base = processor(
        text=[text],
        images=[[image_a, image_b]],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # Tokenize the two possible preference responses separately.
    chosen = processor(
        text=[text + chosen_answer],
        images=[[image_a, image_b]],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    rejected = processor(
        text=[text + rejected_answer],
        images=[[image_a, image_b]],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    prompt_len = base["input_ids"].shape[1]

    def prepare(item):
        ids = item["input_ids"][0]
        labels = ids.clone()
        labels[:prompt_len] = -100
        return {
            "input_ids": ids,
            "attention_mask": item["attention_mask"][0],
            "labels": labels,
            "pixel_values": item.get("pixel_values"),
            "aspect_ratio_ids": item.get("aspect_ratio_ids"),
            "aspect_ratio_mask": item.get("aspect_ratio_mask"),
        }

    return {
        "chosen": prepare(chosen),
        "rejected": prepare(rejected),
    }


# ============================================================
# BLOCK 2 — DATA PREPARATION
# ============================================================

def load_existing_dataset():
    
    print(f"Loading: {DATASET_NAME}")
    dataset = load_dataset(DATASET_PATH, split="train")

    required = {"prompt", "chosen_image", "rejected_image"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Missing required dataset columns: {sorted(missing)}"
        )

    return dataset


def expand_rejected_pairs(dataset):
    """
    Convert each chosen image and its rejected-image set into standard
    one-chosen/one-rejected DPO pairs.
    """
    records = []

    for row in dataset:
        rejected = row["rejected_image"]
        if isinstance(rejected, str):
            rejected = [rejected]

        rejected = rejected[:NUM_REJECTED_PER_CHOSEN]

        for rejected_image in rejected:
            records.append(
                {
                    "prompt": row["prompt"],
                    "chosen_image": row["chosen_image"],
                    "rejected_image": rejected_image,
                }
            )

    if not records:
        raise RuntimeError("No DPO preference pairs were found.")

    return records


class MultimodalDPOCollator:
    """Tokenizes Prompt + two candidate images for pairwise DPO."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        chosen_items = []
        rejected_items = []

        for f in features:
            chosen = load_rgb(f["chosen_image"])
            rejected = load_rgb(f["rejected_image"])

            tokenized = tokenize_example(
                self.processor,
                f["prompt"],
                chosen,
                rejected,
            )

            chosen_items.append(tokenized["chosen"])
            rejected_items.append(tokenized["rejected"])

        return {
            "chosen": chosen_items,
            "rejected": rejected_items,
        }


# ============================================================
# BLOCK 1 — MODEL PREPARATION
# ============================================================

def build_model():
    print("Loading Llama 3.2 Vision...")

    quantization_config = None
    if USE_4BIT and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_use_double_quant=True,
        )

    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Frozen pretrained backbone.
    for p in model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ============================================================
# BLOCK 3 — DPO TRAINER
# ============================================================

def sequence_logprob(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        pixel_values=batch["pixel_values"],
        aspect_ratio_ids=batch.get("aspect_ratio_ids"),
        aspect_ratio_mask=batch.get("aspect_ratio_mask"),
    )

    logits = outputs.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    safe_labels = labels.clamp_min(0)
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)

    mask = labels != -100
    return (token_log_probs * mask).sum(dim=-1)


def stack_batch(items, processor, device):
    ids = pad_sequence(
        [x["input_ids"] for x in items],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id,
    )
    masks = pad_sequence(
        [x["attention_mask"] for x in items],
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        [x["labels"] for x in items],
        batch_first=True,
        padding_value=-100,
    )

    result = {
        "input_ids": ids.to(device),
        "attention_mask": masks.to(device),
        "labels": labels.to(device),
    }

    # Images are equal in shape within a batch because the processor
    # normalizes the candidate image tensors.
    for key in ("pixel_values", "aspect_ratio_ids", "aspect_ratio_mask"):
        values = [x[key] for x in items if x.get(key) is not None]
        if values:
            result[key] = torch.cat(values, dim=0).to(device)

    return result


class PixelSheetDPOTrainer(DPOTrainer):
    """
    Custom multimodal DPO trainer.

    The policy compares the probability of the correct candidate label
    against the incorrect candidate label for the same prompt containing
    both candidate images.
    """

    def __init__(self, processor, *args, **kwargs):
        self.pixel_processor = processor
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device

        chosen = stack_batch(
            inputs["chosen"], self.pixel_processor, device
        )
        rejected = stack_batch(
            inputs["rejected"], self.pixel_processor, device
        )

        chosen_logp = sequence_logprob(model, chosen)
        rejected_logp = sequence_logprob(model, rejected)

        # Frozen reference policy: no gradients.
        with torch.no_grad():
            ref_model = self.ref_model
            ref_chosen_logp = sequence_logprob(ref_model, chosen)
            ref_rejected_logp = sequence_logprob(ref_model, rejected)

        policy_margin = chosen_logp - rejected_logp
        reference_margin = ref_chosen_logp - ref_rejected_logp

        logits = BETA * (policy_margin - reference_margin)
        loss = -torch.nn.functional.logsigmoid(logits).mean()

        if return_outputs:
            return loss, {
                "chosen_logp": chosen_logp.detach(),
                "rejected_logp": rejected_logp.detach(),
            }

        return loss


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

def build_training_args():
    return DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=BETA,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_length=MAX_LENGTH,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",
        bf16=DTYPE == torch.bfloat16 and torch.cuda.is_available(),
        fp16=DTYPE == torch.float16 and torch.cuda.is_available(),
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def train():
    seed_everything(SEED)

    print("=" * 70)
    print("LLAMA 3.2 VISION — PIXEL-SHEET DPO POST-TRAINING")
    print("=" * 70)

    # BLOCK 1 — MODEL PREPARATION
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = build_model()

    # BLOCK 2 — DATA PREPARATION
    dataset = load_existing_dataset()
    preference_pairs = expand_rejected_pairs(dataset)

    print(f"Original preference records: {len(dataset)}")
    print(f"Expanded DPO pairs: {len(preference_pairs)}")
    print(f"NUM_REJECTED_PER_CHOSEN: {NUM_REJECTED_PER_CHOSEN}")

    collator = MultimodalDPOCollator(processor)

    # A frozen reference model is required for DPO.
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    # BLOCK 3 — DPO TRAINING
    trainer = PixelSheetDPOTrainer(
        processor=processor,
        model=model,
        ref_model=ref_model,
        args=build_training_args(),
        train_dataset=preference_pairs,
        data_collator=collator,
    )

    print("Starting DPO training...")
    trainer.train()

    # BLOCK 4 — ADAPTED MODEL
    print("Saving adapted LoRA model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("=" * 70)
    print("DPO POST-TRAINING COMPLETED")
    print(f"Adapter: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    train()
