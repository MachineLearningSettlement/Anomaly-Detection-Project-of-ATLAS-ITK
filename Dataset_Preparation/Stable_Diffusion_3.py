"""
=======================================================================
SD3 SYNTHETIC SEMICONDUCTOR ANOMALY DATASET GENERATION
=======================================================================

Purpose
-------
Generate synthetic semiconductor pixel-sheet anomaly images using:

    PRETRAINED Stable Diffusion 3
            +
    REAL NORMAL PIXEL-SHEET IMAGES
            +
    TEXT PROMPTS

The real images are used as visual references through an IP-Adapter.
The text prompt specifies the anomaly that should be synthesized.

Three anomaly classes:

    1. Chemical Contamination
    2. Dust
    3. Scratch

The real reference images should be NORMAL copper pixel-sheet images.

They provide SD3 with examples of:
    - copper pixel-sheet appearance
    - copper color/material
    - pixel-sheet geometry
    - different real shapes
    - black background
    - real image composition
    - realistic industrial imaging characteristics

The text prompt specifies:
    - the anomaly type
    - realistic semiconductor inspection appearance
    - copper pixel-sheet
    - black background
    - microscopic/industrial appearance

Output
------

synthetic_anomalies/
│
├── Chemical Contamination/
│   ├── chemical_contamination_0001.png
│   ├── chemical_contamination_0002.png
│   └── ...
│
├── Dust/
│   ├── dust_0001.png
│   ├── dust_0002.png
│   └── ...
│
└── Scratch/
    ├── scratch_0001.png
    ├── scratch_0002.png
    └── ...

=======================================================================
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================

import gc
import random
from pathlib import Path

import torch

from PIL import Image

from diffusers import StableDiffusion3Pipeline

from transformers import (
    SiglipVisionModel,
    SiglipImageProcessor,
)


# =====================================================================
# 2. CONFIGURATION
# =====================================================================

# ---------------------------------------------------------------------
# PRETRAINED SD3 MODEL
# ---------------------------------------------------------------------

MODEL_ID = (
    "stabilityai/"
    "stable-diffusion-3-medium-diffusers"
)


# ---------------------------------------------------------------------
# IMAGE REFERENCE DIRECTORY
#
# Put your REAL NORMAL copper pixel-sheet images here.
#
# Example:
#
# real_normal_pixel_sheets/
#     normal_001.png
#     normal_002.png
#     normal_003.png
#     ...
# ---------------------------------------------------------------------

REFERENCE_IMAGES_ROOT = Path(
    "real_normal_pixel_sheets"
)


# ---------------------------------------------------------------------
# OUTPUT DIRECTORY
# ---------------------------------------------------------------------

OUTPUT_ROOT = Path(
    "synthetic_anomalies"
)


# ---------------------------------------------------------------------
# NUMBER OF SYNTHETIC IMAGES PER ANOMALY
# ---------------------------------------------------------------------

IMAGES_PER_CLASS = 100


# ---------------------------------------------------------------------
# IMAGE SIZE
# ---------------------------------------------------------------------

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024


# ---------------------------------------------------------------------
# SD3 GENERATION PARAMETERS
# ---------------------------------------------------------------------

NUM_INFERENCE_STEPS = 28

GUIDANCE_SCALE = 5.0


# ---------------------------------------------------------------------
# IP-ADAPTER IMAGE INFLUENCE
#
# Higher value:
#     generated image follows the real reference more strongly.
#
# Lower value:
#     SD3 has more freedom to generate variations.
#
# 0.6 is a reasonable starting point.
# ---------------------------------------------------------------------

IP_ADAPTER_SCALE = 0.60


# ---------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------

BASE_SEED = 12345


# ---------------------------------------------------------------------
# SUPPORTED IMAGE EXTENSIONS
# ---------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


# =====================================================================
# 3. IP-ADAPTER CONFIGURATION
# =====================================================================

"""
SD3 IP-Adapter requires:

    1. An image encoder
    2. An image processor
    3. An IP-Adapter checkpoint

The image encoder extracts visual features from the real normal
pixel-sheet image.

The text prompt remains responsible for describing the anomaly.

Therefore:

REAL IMAGE
    +
TEXT PROMPT
    ↓
SD3
    ↓
SYNTHETIC ANOMALOUS IMAGE
"""

# ---------------------------------------------------------------------
# Image encoder
# ---------------------------------------------------------------------

IMAGE_ENCODER_ID = (
    "google/"
    "siglip-so400m-patch14-384"
)


# ---------------------------------------------------------------------
# SD3 IP-Adapter
#
# This checkpoint is the SD3.5 Large IP-Adapter from InstantX.
# ---------------------------------------------------------------------

IP_ADAPTER_ID = (
    "InstantX/"
    "SD3.5-Large-IP-Adapter"
)


# =====================================================================
# 4. ANOMALY CLASSES
# =====================================================================

ANOMALY_CLASSES = {

    # -----------------------------------------------------------------
    # CHEMICAL CONTAMINATION
    # -----------------------------------------------------------------

    "Chemical Contamination": {

        "folder_name":
            "Chemical Contamination",

        "filename_prefix":
            "chemical_contamination",

        "prompt": (
            "high-resolution industrial microscope image "
            "of a semiconductor pixel sheet made from copper, "
            "realistic copper pixel-sheet geometry and structure, "
            "dark black background, "
            "real semiconductor manufacturing inspection image, "
            "microscopic copper surface, "
            "visible localized chemical contamination on the "
            "copper pixel sheet, "
            "irregular chemical residue, "
            "subtle chemical stains and contamination marks, "
            "realistic contamination morphology, "
            "natural irregular boundaries, "
            "physically plausible manufacturing defect, "
            "preserve the original pixel-sheet geometry and shape, "
            "photorealistic, "
            "scientific imaging, "
            "industrial quality inspection photograph, "
            "sharp microscopic details"
        ),
    },


    # -----------------------------------------------------------------
    # DUST
    # -----------------------------------------------------------------

    "Dust": {

        "folder_name":
            "Dust",

        "filename_prefix":
            "dust",

        "prompt": (
            "high-resolution industrial microscope image "
            "of a semiconductor pixel sheet made from copper, "
            "realistic copper pixel-sheet geometry and structure, "
            "dark black background, "
            "real semiconductor manufacturing inspection image, "
            "microscopic copper surface, "
            "visible dust contamination on the copper pixel sheet, "
            "small irregular dust particles and microscopic specks, "
            "naturally distributed dust particles, "
            "different realistic particle sizes, "
            "subtle shadows around dust particles, "
            "physically plausible manufacturing defect, "
            "preserve the original pixel-sheet geometry and shape, "
            "photorealistic, "
            "scientific imaging, "
            "industrial quality inspection photograph, "
            "sharp microscopic details"
        ),
    },


    # -----------------------------------------------------------------
    # SCRATCH
    # -----------------------------------------------------------------

    "Scratch": {

        "folder_name":
            "Scratch",

        "filename_prefix":
            "scratch",

        "prompt": (
            "high-resolution industrial microscope image "
            "of a semiconductor pixel sheet made from copper, "
            "realistic copper pixel-sheet geometry and structure, "
            "dark black background, "
            "real semiconductor manufacturing inspection image, "
            "microscopic copper surface, "
            "visible physical scratch on the copper pixel sheet, "
            "thin irregular scratch extending across the surface, "
            "realistic surface damage, "
            "naturally varying scratch width and depth, "
            "subtle damaged copper texture around the scratch, "
            "physically plausible manufacturing defect, "
            "preserve the original pixel-sheet geometry and shape, "
            "photorealistic, "
            "scientific imaging, "
            "industrial quality inspection photograph, "
            "sharp microscopic details"
        ),
    },
}


# =====================================================================
# 5. NEGATIVE PROMPT
# =====================================================================

NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, anime, CGI, "
    "3D render, fantasy, landscape, person, human, face, "
    "animal, building, text, letters, numbers, logo, watermark, "
    "colored background, white background, "
    "blue background, green background, "
    "unrealistic copper, plastic, glass, "
    "completely different object, "
    "different geometry, "
    "distorted pixel sheet, "
    "deformed pixel sheet, "
    "low resolution, blurry, "
    "extreme noise, oversaturated, "
    "unrealistic defect, "
    "multiple unrelated objects"
)


# =====================================================================
# 6. DEVICE
# =====================================================================

def get_device():
    """
    Detect the available computation device.
    """

    if torch.cuda.is_available():

        return "cuda"

    return "cpu"


DEVICE = get_device()


# =====================================================================
# 7. CREATE OUTPUT DIRECTORIES
# =====================================================================

def create_output_directories():

    """
    Create:

        synthetic_anomalies/
            Chemical Contamination/
            Dust/
            Scratch/
    """

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    for anomaly_name, config in ANOMALY_CLASSES.items():

        folder = (
            OUTPUT_ROOT
            / config["folder_name"]
        )

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"[INFO] Output directory: {folder}"
        )


# =====================================================================
# 8. FIND REAL REFERENCE IMAGES
# =====================================================================

def get_reference_images():

    """
    Find all real normal pixel-sheet images.

    These images are NOT anomaly images.

    They are examples of the real pixel-sheet geometry,
    copper material and black-background appearance.
    """

    if not REFERENCE_IMAGES_ROOT.exists():

        raise FileNotFoundError(
            "\nReal reference-image directory does not exist:\n"
            f"{REFERENCE_IMAGES_ROOT.resolve()}\n\n"
            "Create this directory and put your normal "
            "pixel-sheet images inside it."
        )

    reference_images = []

    for path in REFERENCE_IMAGES_ROOT.iterdir():

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            reference_images.append(path)

    if len(reference_images) == 0:

        raise RuntimeError(
            "\nNo real reference images were found in:\n"
            f"{REFERENCE_IMAGES_ROOT.resolve()}\n\n"
            "Put normal copper pixel-sheet images "
            "inside this directory."
        )

    reference_images.sort()

    print(
        f"[INFO] Found "
        f"{len(reference_images)} "
        f"real reference images."
    )

    return reference_images


# =====================================================================
# 9. LOAD IMAGE ENCODER
# =====================================================================

def load_image_encoder():

    """
    Load the pretrained SigLIP vision encoder used by the
    SD3 IP-Adapter.

    The encoder extracts visual information from the real
    pixel-sheet reference image.
    """

    print("\n")
    print("=" * 70)
    print("Loading Image Encoder")
    print("=" * 70)

    print(
        f"[INFO] Image encoder: "
        f"{IMAGE_ENCODER_ID}"
    )

    if DEVICE == "cuda":

        image_processor = (
            SiglipImageProcessor.from_pretrained(
                IMAGE_ENCODER_ID
            )
        )

        image_encoder = (
            SiglipVisionModel.from_pretrained(
                IMAGE_ENCODER_ID,
                torch_dtype=torch.float16
            )
        )

        image_encoder = (
            image_encoder.to("cuda")
        )

    else:

        image_processor = (
            SiglipImageProcessor.from_pretrained(
                IMAGE_ENCODER_ID
            )
        )

        image_encoder = (
            SiglipVisionModel.from_pretrained(
                IMAGE_ENCODER_ID
            )
        )

        image_encoder = (
            image_encoder.to("cpu")
        )

    print(
        "[INFO] Image encoder loaded."
    )

    return (
        image_encoder,
        image_processor
    )


# =====================================================================
# 10. LOAD PRETRAINED SD3 + IP-ADAPTER
# =====================================================================

def load_sd3_pipeline(
    image_encoder,
    image_processor
):

    """
    Load pretrained SD3 and connect the IP-Adapter.

    No SD3 fine-tuning is performed.

    The model remains pretrained.
    """

    print("\n")
    print("=" * 70)
    print("Loading Stable Diffusion 3")
    print("=" * 70)

    print(
        f"[INFO] Model: {MODEL_ID}"
    )

    print(
        f"[INFO] Device: {DEVICE}"
    )

    if DEVICE == "cuda":

        pipe = (
            StableDiffusion3Pipeline
            .from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                image_encoder=image_encoder,
                feature_extractor=image_processor,
            )
        )

        # -------------------------------------------------------------
        # Reduce GPU memory usage.
        # -------------------------------------------------------------

        pipe.enable_model_cpu_offload()

    else:

        print(
            "[WARNING] CUDA GPU not detected."
        )

        print(
            "[WARNING] SD3 generation on CPU "
            "will be extremely slow."
        )

        pipe = (
            StableDiffusion3Pipeline
            .from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                image_encoder=image_encoder,
                feature_extractor=image_processor,
            )
        )

        pipe.to("cpu")

    # -----------------------------------------------------------------
    # Load the pretrained IP-Adapter.
    # -----------------------------------------------------------------

    print(
        "[INFO] Loading SD3 IP-Adapter..."
    )

    pipe.load_ip_adapter(
        IP_ADAPTER_ID
    )

    # -----------------------------------------------------------------
    # Set image-reference influence.
    # -----------------------------------------------------------------

    pipe.set_ip_adapter_scale(
        IP_ADAPTER_SCALE
    )

    print(
        f"[INFO] IP-Adapter scale: "
        f"{IP_ADAPTER_SCALE}"
    )

    print(
        "[INFO] SD3 + IP-Adapter loaded successfully."
    )

    return pipe


# =====================================================================
# 11. LOAD REFERENCE IMAGE
# =====================================================================

def load_reference_image(
    image_path
):

    """
    Load a real normal pixel-sheet image.

    This image is used as visual guidance for SD3.

    It is NOT copied directly into the output.

    SD3 receives its visual features through the IP-Adapter.
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    return image


# =====================================================================
# 12. GENERATE ONE IMAGE
# =====================================================================

def generate_image(
    pipe,
    reference_image,
    prompt,
    seed
):

    """
    Generate one anomalous image using:

        REAL REFERENCE IMAGE
                    +
              TEXT PROMPT
                    ↓
                   SD3
                    ↓
        SYNTHETIC ANOMALOUS IMAGE

    The reference image controls the visual characteristics,
    while the text prompt specifies the anomaly.
    """

    generator = (
        torch.Generator(
            device="cpu"
        ).manual_seed(seed)
    )

    result = pipe(

        # -------------------------------------------------------------
        # TEXT CONDITIONING
        # -------------------------------------------------------------

        prompt=prompt,

        # -------------------------------------------------------------
        # NEGATIVE CONDITIONING
        # -------------------------------------------------------------

        negative_prompt=NEGATIVE_PROMPT,

        # -------------------------------------------------------------
        # REAL IMAGE CONDITIONING
        # -------------------------------------------------------------

        ip_adapter_image=reference_image,

        # -------------------------------------------------------------
        # OUTPUT SIZE
        # -------------------------------------------------------------

        height=IMAGE_HEIGHT,

        width=IMAGE_WIDTH,

        # -------------------------------------------------------------
        # GENERATION PARAMETERS
        # -------------------------------------------------------------

        num_inference_steps=(
            NUM_INFERENCE_STEPS
        ),

        guidance_scale=(
            GUIDANCE_SCALE
        ),

        generator=generator,
    )

    return result.images[0]


# =====================================================================
# 13. GENERATE ONE ANOMALY CLASS
# =====================================================================

def generate_anomaly_class(
    pipe,
    anomaly_name,
    config,
    reference_images
):

    """
    Generate all images belonging to one anomaly class.

    Each generated image randomly selects one real normal
    pixel-sheet image as the visual reference.

    This is important because the real dataset may contain
    many different pixel-sheet shapes.

    Therefore:

        Real Shape A → synthetic anomaly
        Real Shape B → synthetic anomaly
        Real Shape C → synthetic anomaly
        ...

    This preserves diversity across the synthetic dataset.
    """

    folder = (
        OUTPUT_ROOT
        / config["folder_name"]
    )

    prefix = (
        config["filename_prefix"]
    )

    prompt = (
        config["prompt"]
    )

    print("\n")
    print("=" * 70)
    print(
        f"Generating: {anomaly_name}"
    )
    print("=" * 70)

    print(
        f"[INFO] Images requested: "
        f"{IMAGES_PER_CLASS}"
    )

    for index in range(
        1,
        IMAGES_PER_CLASS + 1
    ):

        # -------------------------------------------------------------
        # Select a real normal image.
        #
        # Random selection gives the synthetic dataset access to
        # different real pixel-sheet shapes.
        # -------------------------------------------------------------

        reference_path = random.choice(
            reference_images
        )

        print(
            f"\n[{index:04d}/"
            f"{IMAGES_PER_CLASS:04d}] "
            f"Generating {anomaly_name}"
        )

        print(
            f"[INFO] Reference: "
            f"{reference_path.name}"
        )

        # -------------------------------------------------------------
        # Unique reproducible seed.
        # -------------------------------------------------------------

        seed = (
            BASE_SEED
            + index
            + (
                list(ANOMALY_CLASSES.keys())
                .index(anomaly_name)
                * 100000
            )
        )

        print(
            f"[INFO] Seed: {seed}"
        )

        try:

            # ---------------------------------------------------------
            # Load real normal image.
            # ---------------------------------------------------------

            reference_image = (
                load_reference_image(
                    reference_path
                )
            )

            # ---------------------------------------------------------
            # Generate anomalous image.
            # ---------------------------------------------------------

            synthetic_image = (
                generate_image(
                    pipe=pipe,
                    reference_image=reference_image,
                    prompt=prompt,
                    seed=seed
                )
            )

            # ---------------------------------------------------------
            # Output filename.
            # ---------------------------------------------------------

            filename = (
                f"{prefix}_"
                f"{index:04d}.png"
            )

            output_path = (
                folder
                / filename
            )

            # ---------------------------------------------------------
            # Save.
            # ---------------------------------------------------------

            synthetic_image.save(
                output_path,
                format="PNG"
            )

            print(
                f"[SUCCESS] Saved: "
                f"{output_path}"
            )

            # ---------------------------------------------------------
            # Free memory.
            # ---------------------------------------------------------

            del (
                reference_image,
                synthetic_image
            )

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        except Exception as error:

            print(
                f"[ERROR] Failed to generate "
                f"{anomaly_name} image "
                f"{index}."
            )

            print(
                f"[ERROR] {error}"
            )

            # ---------------------------------------------------------
            # Continue with next image.
            # ---------------------------------------------------------

            continue


# =====================================================================
# 14. GENERATE COMPLETE DATASET
# =====================================================================

def generate_dataset(
    pipe,
    reference_images
):

    """
    Generate:

        Chemical Contamination
        Dust
        Scratch
    """

    print("\n")
    print("=" * 70)
    print("STARTING SYNTHETIC DATASET GENERATION")
    print("=" * 70)

    total_requested = (
        len(ANOMALY_CLASSES)
        * IMAGES_PER_CLASS
    )

    print(
        f"[INFO] Number of real references: "
        f"{len(reference_images)}"
    )

    print(
        f"[INFO] Number of anomaly classes: "
        f"{len(ANOMALY_CLASSES)}"
    )

    print(
        f"[INFO] Images per class: "
        f"{IMAGES_PER_CLASS}"
    )

    print(
        f"[INFO] Total requested images: "
        f"{total_requested}"
    )

    # -----------------------------------------------------------------
    # Generate each anomaly class.
    # -----------------------------------------------------------------

    for anomaly_name, config in (
        ANOMALY_CLASSES.items()
    ):

        generate_anomaly_class(

            pipe=pipe,

            anomaly_name=anomaly_name,

            config=config,

            reference_images=reference_images
        )


# =====================================================================
# 15. DATASET SUMMARY
# =====================================================================

def print_dataset_summary():

    """
    Print the number of generated images in every class.
    """

    print("\n")
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    total_images = 0

    for anomaly_name, config in (
        ANOMALY_CLASSES.items()
    ):

        folder = (
            OUTPUT_ROOT
            / config["folder_name"]
        )

        images = list(
            folder.glob("*.png")
        )

        count = len(images)

        total_images += count

        print(
            f"{anomaly_name:<25}"
            f": {count} images"
        )

    print("-" * 70)

    print(
        f"{'TOTAL':<25}"
        f": {total_images} images"
    )

    print("-" * 70)

    print(
        f"Output directory:\n"
        f"{OUTPUT_ROOT.resolve()}"
    )


# =====================================================================
# 16. MAIN
# =====================================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "SD3 SEMICONDUCTOR ANOMALY SYNTHESIS"
    )
    print("=" * 70)

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    print(
        f"\n[INFO] Model:"
    )

    print(
        f"       {MODEL_ID}"
    )

    print(
        f"\n[INFO] Device:"
    )

    print(
        f"       {DEVICE}"
    )

    print(
        f"\n[INFO] Reference images:"
    )

    print(
        f"       "
        f"{REFERENCE_IMAGES_ROOT.resolve()}"
    )

    print(
        f"\n[INFO] Output:"
    )

    print(
        f"       "
        f"{OUTPUT_ROOT.resolve()}"
    )

    print(
        f"\n[INFO] IP-Adapter scale:"
    )

    print(
        f"       {IP_ADAPTER_SCALE}"
    )

    # -----------------------------------------------------------------
    # Create output folders.
    # -----------------------------------------------------------------

    create_output_directories()

    # -----------------------------------------------------------------
    # Find real normal images.
    # -----------------------------------------------------------------

    reference_images = (
        get_reference_images()
    )

    # -----------------------------------------------------------------
    # Load image encoder.
    # -----------------------------------------------------------------

    (
        image_encoder,
        image_processor
    ) = load_image_encoder()

    # -----------------------------------------------------------------
    # Load SD3 + IP-Adapter.
    # -----------------------------------------------------------------

    pipe = load_sd3_pipeline(
        image_encoder=image_encoder,
        image_processor=image_processor
    )

    # -----------------------------------------------------------------
    # Generate synthetic dataset.
    # -----------------------------------------------------------------

    generate_dataset(
        pipe=pipe,
        reference_images=reference_images
    )

    # -----------------------------------------------------------------
    # Print summary.
    # -----------------------------------------------------------------

    print_dataset_summary()

    # -----------------------------------------------------------------
    # Release memory.
    # -----------------------------------------------------------------

    del pipe
    del image_encoder
    del image_processor

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    print("\n")
    print("=" * 70)
    print("GENERATION COMPLETED")
    print("=" * 70)


# =====================================================================
# 17. PROGRAM ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()
