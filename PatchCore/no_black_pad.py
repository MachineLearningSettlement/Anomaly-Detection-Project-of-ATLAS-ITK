import os
from PIL import Image
import numpy as np

# Chemin vers le dossier contenant les images
dossier = "/local/home/oe283118/Bureau/AtlasITKDataSets/dataset_images_new_modules_resize128/maindata_resize128/train/good"

# Seuil de suppression en pourcentage 
seuil_noir = 0.9       # 90% as best one

# Parcourir toutes les images .png du dossier
for filename in os.listdir(dossier):
    if filename.lower().endswith(".png"):
        chemin_image = os.path.join(dossier, filename)

        # Ouvrir l'image et la convertir en niveaux de gris
        with Image.open(chemin_image).convert("L") as img:
            # Convertir en array numpy
            img_array = np.array(img)

            # Calcul du pourcentage de pixels noirs (valeur <= 10/255 pour tolérance)
            pixels_noirs = np.sum(img_array <= 10)
            total_pixels = img_array.size
            pourcentage_noir = pixels_noirs / total_pixels

            # Suppression si seuil dépassé
            if pourcentage_noir >= seuil_noir:
                os.remove(chemin_image)
                print(f"Supprimée : {filename} ({pourcentage_noir:.2%} de pixels noirs)")
