import os
from PIL import Image
import numpy as np

# Paramètre modifiable : taille de la bordure à noircir
param = 12

# Dossiers source et destination
input_root = "/feynman/home/dedip/lemid/oe283118/work/results/Patchcore/itkPolar_ParisPreProd_128_clean_brut_resize2_70_copie_tresholdtiles"
output_root = "/feynman/home/dedip/lemid/oe283118/work/results/Patchcore/itkPolar_ParisPreProd_128_clean_brut_resize2_70_copie_erodingtiles"

# Sous-dossiers à traiter
subfolders = ['chemical_contamination', 'dust', 'scratch']

os.makedirs(output_root, exist_ok=True)

for subfolder in subfolders:
    input_folder = os.path.join(input_root, subfolder)
    output_folder = os.path.join(output_root, subfolder)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Charger l'image en niveaux de gris
            img = Image.open(input_path).convert('L')
            arr = np.array(img)

            # Érosion manuelle des bordures
            arr[:param, :] = 0              # haut
            arr[-param:, :] = 0             # bas
            arr[:, :param] = 0              # gauche
            arr[:, -param:] = 0             # droite

            # Sauvegarder l'image modifiée
            eroded_img = Image.fromarray(arr)
            eroded_img.save(output_path)

print(f"Masques érodés sauvegardés dans '{output_root}'")
