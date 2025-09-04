import os
from PIL import Image
import numpy as np

# Chemins vers les deux dossiers racines
root1 = "/feynman/home/dedip/lemid/oe283118/work/results/Patchcore/itkPolar_ParisPreProd_128_clean_brut_resize2_70_copie_erodingtiles"
root2 = "/feynman/home/dedip/lemid/oe283118/work/results/Patchcore/itkPolar_ParisPreProd_128_clean_brut_resize2_70_copie_output_masks"
output_root = "/feynman/home/dedip/lemid/oe283118/work/results/Patchcore/itkPolar_ParisPreProd_128_clean_brut_resize2_70_copie_output_mask_intersection"

# Sous-dossiers à traiter
subfolders = ["chemical_contamination", "dust", "scratch"]

# Création du dossier de sortie
os.makedirs(output_root, exist_ok=True)

for subfolder in subfolders:
    folder1 = os.path.join(root1, subfolder)
    folder2 = os.path.join(root2, subfolder)
    output_folder = os.path.join(output_root, subfolder)
    os.makedirs(output_folder, exist_ok=True)

    # Lister les fichiers triés (pour s’assurer de la correspondance)
    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))

    for f1, f2 in zip(files1, files2):
        path1 = os.path.join(folder1, f1)
        path2 = os.path.join(folder2, f2)
        output_path = os.path.join(output_folder, f1)  # on garde le nom de f1

        # Charger les deux images en niveaux de gris
        img1 = Image.open(path1).convert("L")
        img2 = Image.open(path2).convert("L").resize((128, 128), Image.NEAREST)  
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        # Intersection logique : garder blanc uniquement si blanc dans les deux
        intersection = np.where((arr1 == 255) & (arr2 == 255), 255, 0).astype(np.uint8)

        # Sauvegarder l'image résultante
        Image.fromarray(intersection).save(output_path)

print("Intersections binaires créées et enregistrées dans 'output/'")
