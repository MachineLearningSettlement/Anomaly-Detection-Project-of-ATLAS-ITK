import os
from PIL import Image
import numpy as np


# Dossier d'entrée
input_root = "/feynman/home/dedip/lemid/oe283118/work/anomalib/datasets/MVTecAD/itkPolar_ParisPreProd_128_clean_brut_resize2_70_copie/test"
output_root = "/feynman/home/dedip/lemid/oe283118/work/results/Patchcore/itkPolar_ParisPreProd_128_clean_brut_resize2_70_copie_tresholdtiles"

# Créer le dossier output s'il n'existe pas
os.makedirs(output_root, exist_ok=True)

# Sous-dossiers à traiter
subfolders = ['chemical_contamination', 'dust', 'scratch']

for subfolder in subfolders:
    input_folder = os.path.join(input_root, subfolder)
    output_folder = os.path.join(output_root, subfolder)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Ouvrir et convertir en image RGB
            image = Image.open(input_path).convert('RGB')
            np_image = np.array(image)

            # Créer un masque : tout sauf le noir devient blanc
            mask = np.where(np.all(np_image == [0, 0, 0], axis=-1), 0, 255).astype(np.uint8)

            # Convertir en image (1 canal)
            mask_image = Image.fromarray(mask, mode='L')
            mask_image.save(output_path)

print(" Masques générés et enregistrés dans le dossier 'output'")
