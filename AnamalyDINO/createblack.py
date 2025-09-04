import os
from PIL import Image
import numpy as np

def create_black_images_like_source(source_path, target_path, size=(128, 128)):
    # Créer le dossier cible s'il n'existe pas
    os.makedirs(target_path, exist_ok=True)
    
    # Lister tous les fichiers dans le dossier source
    image_filenames = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]

    for filename in image_filenames:
        # Créer une image noire
        black_image = Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8))  # RGB
        # Construire le chemin d'enregistrement
        save_path = os.path.join(target_path, filename)
        # Enregistrer l'image noire avec le même nom
        black_image.save(save_path)

    print(f"{len(image_filenames)} images noires enregistrées dans : {target_path}")

# Exemple d'utilisation

source_path = "/feynman/home/dedip/lemid/oe283118/work/anomalib/datasets/MVTecAD/dataset_original_size_256_AnomalyDINO/test/good"

target_path = "/feynman/home/dedip/lemid/oe283118/work/AnomalyDINO/results/dataset_original_size_256_AnomalyDINO_masks_gth/good"

create_black_images_like_source(source_path, target_path)
