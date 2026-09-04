"""
Llama 3.2 Vision — DPO post-training for pixel-sheet retrieval.

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
    with each chosen image. Each chosen/rejected pair becomes one DPO sample.

IMPORTANT IMPLEMENTATION DETAIL
    Llama 3.2 Vision is a vision-language model whose output is text.
    Therefore, the DPO formulation presents BOTH candidate images in the
    same multimodal prompt and trains the model to prefer the candidate
    corresponding to the chosen image ("Image 1" or "Image 2").

    TRL's native DataCollatorForVisionPreference performs the multimodal
    image processing and tokenization, while DPOTrainer performs the
    standard chosen-vs-rejected preference optimization.
"""

import random
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from trl.trainer.dpo_trainer import DataCollatorForVisionPreference


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

DATASET_NAME = "Llama3.2_Vision_DPO_PixelSheet_Retrieval"
DATASET_PATH = "llama32_vision_dpo_pixel_sheet_retrieval"

# Hyperparameter requested for rejected-image selection.
NUM_REJECTED_PER_CHOSEN = 5

BETA = 0.1
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 5e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_LENGTH = None

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


def build_preference_record(prompt, chosen_image_path, rejected_image_path):
    """
    Build one multimodal DPO preference sample.

    Both candidate images are presented to Llama 3.2 Vision as inputs.
    The DPO responses are the candidate selections ("Image 1" / "Image 2").
    Candidate order is randomized so the model cannot learn a positional
    shortcut such as always preferring Image 1.
    """

    chosen_image = load_rgb(chosen_image_path)
    rejected_image = load_rgb(rejected_image_path)

    chosen_first = random.random() < 0.5

    if chosen_first:
        images = [chosen_image, rejected_image]
        chosen_answer = "Image 1"
        rejected_answer = "Image 2"
    else:
        images = [rejected_image, chosen_image]
        chosen_answer = "Image 2"
        rejected_answer = "Image 1"

    return {
        "images": images,

        "prompt": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            prompt
                            + "\n\n"
                            "Two candidate images are provided. "
                            "They are ordered as Image 1 and Image 2. "
                            "Select the image that best matches the description."
                        ),
                    }
                ],
            }
        ],

        "chosen": [
            {
                "role": "assistant",
                "content": chosen_answer,
            }
        ],

        "rejected": [
            {
                "role": "assistant",
                "content": rejected_answer,
            }
        ],
    }


# ============================================================
# BLOCK 2 — DATA PREPARATION
# ============================================================

def load_existing_dataset():
    """Load the assumed existing dataset; do not create training data."""
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
    multimodal DPO preference samples.

    For every chosen image, NUM_REJECTED_PER_CHOSEN rejected images are
    selected. Each chosen/rejected pair becomes one DPO sample.
    """

    records = []

    for row in dataset:

        rejected = row["rejected_image"]

        if isinstance(rejected, str):
            rejected = [rejected]

        rejected = rejected[:NUM_REJECTED_PER_CHOSEN]

        for rejected_image in rejected:

            records.append(
                build_preference_record(
                    prompt=row["prompt"],
                    chosen_image_path=row["chosen_image"],
                    rejected_image_path=rejected_image,
                )
            )

    if not records:
        raise RuntimeError("No DPO preference pairs were found.")

    return Dataset.from_list(records)


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

def build_dpo_trainer(
    model,
    processor,
    preference_dataset,
):
    """
    Build the native TRL multimodal DPO trainer.

    DataCollatorForVisionPreference processes:
        images + prompt + chosen + rejected

    DPOTrainer uses the initial policy as the frozen reference policy
    when ref_model=None, while the LoRA-adapted model is optimized.
    """

    data_collator = DataCollatorForVisionPreference(
        processor=processor,
    )

    return DPOTrainer(
        model=model,
        ref_model=None,
        args=build_training_args(),
        train_dataset=preference_dataset,
        processing_class=processor,
        data_collator=data_collator,
    )


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
        max_length=MAX_LENGTH,  # Keep None for VLMs to avoid truncating image tokens.
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

    # BLOCK 3 — DPO TRAINING
    trainer = build_dpo_trainer(
        model=model,
        processor=processor,
        preference_dataset=preference_pairs,
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
