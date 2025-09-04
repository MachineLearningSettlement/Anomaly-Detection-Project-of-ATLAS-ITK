import os
from PIL import Image, ImageEnhance

# Répertoire source (et destination)
input_dir = "/local/home/oe283118/Bureau/AtlasITKDataSets/dataset_images_new_modules_lite/maindata/test/scratch"

# Hyperparamètres d'ajustement
contrast_factor = 1.5     # >1 augmente le contraste, <1 le réduit
brightness_factor = 1.2   # >1 rend l'image plus lumineuse, <1 plus sombre

# Boucle sur chaque image
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".png"):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)

        # Ajuster la luminosité
        enhancer_brightness = ImageEnhance.Brightness(img)
        img = enhancer_brightness.enhance(brightness_factor)

        # Ajuster le contraste
        enhancer_contrast = ImageEnhance.Contrast(img)
        img = enhancer_contrast.enhance(contrast_factor)

        # Sauvegarde par-dessus l'image originale
        img.save(img_path)

        print(f"Done: {filename}")
