import gc
import random
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline
from transformers import SiglipVisionModel, SiglipImageProcessor

NORMAL_IMAGES_ROOT = Path("real_normal_pixel_sheets")
ANOMALOUS_IMAGES_ROOT = Path("real_anomalous_pixel_sheets")
OUTPUT_ROOT = Path("synthetic_anomalies")

MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
IMAGE_ENCODER_ID = "google/siglip-so400m-patch14-384"
IP_ADAPTER_ID = "InstantX/SD3.5-Large-IP-Adapter"

IMAGES_PER_CLASS = 150
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 5.0
IMG2IMG_STRENGTH = 0.40
IP_ADAPTER_SCALE = 0.65
BASE_SEED = 12345
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


ANOMALY_PROMPTS = {
    "Chemical Contamination": """High-resolution industrial microscope image of a semiconductor pixel sheet made from copper, realistic copper pixel-sheet geometry, on a pure dark black background. Synthesize a realistic localized chemical contamination defect on the copper pixel sheet. The defect must resemble the morphology and visual characteristics of the provided real chemical-contamination reference image. Include irregular chemical residue, subtle contamination stains and realistic surface changes caused by chemical contamination. Preserve the copper pixel-sheet geometry from the normal base image. The anomaly must look physically present on the copper surface. Photorealistic scientific industrial inspection image, sharp microscopic details, realistic materials and lighting.""",
    "Dust": """High-resolution industrial microscope image of a semiconductor pixel sheet made from copper, realistic copper pixel-sheet geometry, on a pure dark black background. Synthesize realistic dust contamination on the copper pixel sheet. The dust must resemble the morphology and visual characteristics of the provided real dust reference image. Include small irregular dust particles and microscopic specks with natural size variation, distribution and subtle physical shadows. Preserve the copper pixel-sheet geometry from the normal base image. The anomaly must look physically present on the copper surface. Photorealistic scientific industrial inspection image, sharp microscopic details, realistic materials and lighting.""",
    "Scratch": """High-resolution industrial microscope image of a semiconductor pixel sheet made from copper, realistic copper pixel-sheet geometry, on a pure dark black background. Synthesize a realistic physical scratch defect on the copper pixel sheet. The scratch must resemble the morphology and visual characteristics of the provided real scratch reference image. Include a thin irregular scratch line with realistic variation in width and depth, damaged copper texture and physically plausible surface damage. Preserve the copper pixel-sheet geometry from the normal base image. The anomaly must look physically present on the copper surface. Photorealistic scientific industrial inspection image, sharp microscopic details, realistic materials and lighting.""",
}


NEGATIVE_PROMPT = """cartoon, illustration, anime, drawing, CGI, 3D render, fantasy, people, human, face, text, letters, numbers, watermark, logo, colored background, white background, gray background, unrealistic copper, plastic, glass, metal other than copper, different object, different scene, wrong geometry, distorted pixel sheet, deformed pixel sheet, melted pixel sheet, missing pixel sheet, low resolution, blurry, noisy, unrealistic defect, artificial-looking anomaly, oversized anomaly, excessive damage, duplicate defects, unrealistic dust, unrealistic scratch, unrealistic contamination"""


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_output_directories():
    for anomaly_class in ANOMALY_PROMPTS:
        (OUTPUT_ROOT / anomaly_class).mkdir(parents=True, exist_ok=True)


def get_images(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"Reference directory does not exist: {root}")
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise RuntimeError(f"No reference images found in: {root}")
    return sorted(images)


def get_anomalous_images(anomaly_class: str):
    anomaly_dir = ANOMALOUS_IMAGES_ROOT / anomaly_class
    if not anomaly_dir.exists():
        raise FileNotFoundError(f"Missing anomalous-reference directory: {anomaly_dir}")
    images = [p for p in anomaly_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise RuntimeError(f"No anomalous reference images found for {anomaly_class}: {anomaly_dir}")
    return sorted(images)


def load_reference_image(path: Path):
    return Image.open(path).convert("RGB")


def load_image_encoder(device):
    dtype = torch.float16 if device == "cuda" else torch.float32
    feature_extractor = SiglipImageProcessor.from_pretrained(IMAGE_ENCODER_ID)
    image_encoder = SiglipVisionModel.from_pretrained(IMAGE_ENCODER_ID, torch_dtype=dtype)
    if device == "cuda":
        image_encoder = image_encoder.to(device)
    return image_encoder, feature_extractor


def load_sd3_pipeline(image_encoder, feature_extractor, device):
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
    )
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    pipe.load_ip_adapter(IP_ADAPTER_ID)
    pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
    return pipe


def generate_image(pipe, normal_reference, anomalous_reference, prompt, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        # NORMAL IMAGE: base/geometry/shape of the copper pixel sheet
        image=normal_reference,
        # REAL ANOMALOUS IMAGE: visual/morphological reference for the target defect
        ip_adapter_image=anomalous_reference,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        strength=IMG2IMG_STRENGTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    )
    return result.images[0]


def generate_anomaly_class(pipe, anomaly_class, normal_reference_paths, anomalous_reference_paths):
    output_dir = OUTPUT_ROOT / anomaly_class
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = ANOMALY_PROMPTS[anomaly_class]

    for index in range(IMAGES_PER_CLASS):
        normal_path = random.choice(normal_reference_paths)
        anomalous_path = random.choice(anomalous_reference_paths)
        normal_reference = load_reference_image(normal_path)
        anomalous_reference = load_reference_image(anomalous_path)
        seed = BASE_SEED + index

        print(f"[{index + 1}/{IMAGES_PER_CLASS}] {anomaly_class} | normal={normal_path.name} | anomaly={anomalous_path.name}")

        image = generate_image(
            pipe=pipe,
            normal_reference=normal_reference,
            anomalous_reference=anomalous_reference,
            prompt=prompt,
            seed=seed,
        )

        image.save(output_dir / f"{index + 1:04d}.png")
        del normal_reference, anomalous_reference, image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def generate_dataset(pipe, normal_reference_paths):
    for anomaly_class in ANOMALY_PROMPTS:
        anomalous_reference_paths = get_anomalous_images(anomaly_class)
        generate_anomaly_class(pipe, anomaly_class, normal_reference_paths, anomalous_reference_paths)


def print_dataset_summary():
    print("\nDATASET SUMMARY")
    total = 0
    for anomaly_class in ANOMALY_PROMPTS:
        count = len(list((OUTPUT_ROOT / anomaly_class).glob("*.png")))
        print(f"{anomaly_class}: {count} images")
        total += count
    print(f"Total: {total}")


def main():
    print("SD3.5 SYNTHETIC ANOMALY GENERATION")
    print("Normal images = Img2Img base / pixel-sheet geometry")
    print("Anomalous images = IP-Adapter anomaly morphology reference")
    print("Text prompt = requested anomaly + copper pixel sheet + black background")

    device = get_device()
    print(f"Device: {device}")
    create_output_directories()

    normal_reference_paths = get_images(NORMAL_IMAGES_ROOT)
    print(f"Found {len(normal_reference_paths)} normal reference images.")

    for anomaly_class in ANOMALY_PROMPTS:
        paths = get_anomalous_images(anomaly_class)
        print(f"Found {len(paths)} {anomaly_class} reference images.")

    image_encoder, feature_extractor = load_image_encoder(device)
    pipe = load_sd3_pipeline(image_encoder, feature_extractor, device)

    generate_dataset(pipe, normal_reference_paths)
    print_dataset_summary()

    del pipe, image_encoder, feature_extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
