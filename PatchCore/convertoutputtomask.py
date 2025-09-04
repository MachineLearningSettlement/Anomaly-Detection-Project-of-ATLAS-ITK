import cv2
import numpy as np
import os

input_folder = "/feynman/home/dedip/lemid/oe283118/work/results/Patchcore/itkPolar_ParisPreProd_128_clean_brut_resize2_70_copie_output"

output_folder = "/feynman/home/dedip/lemid/oe283118/work/results/Patchcore/itkPolar_ParisPreProd_128_clean_brut_resize2_70_copie_output_masks"


# Parcourir tous les sous-dossiers dans input_folder 
for subfolder in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder)
    if os.path.isdir(subfolder_path):
        # Créer le sous-dossier correspondant dans le dossier de sortie
        output_subfolder = os.path.join(output_folder, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)
        
        # Parcourir toutes les images dans le sous-dossier
        for filename in os.listdir(subfolder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(subfolder_path, filename)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Erreur de lecture : {image_path}")
                    continue
                
                # Convertir en HSV
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                # Définir les plages de rouge (HSV)
                lower_red1 = np.array([0, 70, 50])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 70, 50])
                upper_red2 = np.array([180, 255, 255])

                # Appliquer les deux masques de rouge
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                red_mask = cv2.bitwise_or(mask1, mask2)

                # Trouver les contours rouges
                contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Créer une image noire (une seule couche)
                mask_output = np.zeros_like(image[:, :, 0])

                # Dessiner les contours remplis en blanc
                cv2.drawContours(mask_output, contours, -1, color=255, thickness=cv2.FILLED)

                # Sauvegarder le masque dans le sous-dossier de sortie
                output_path = os.path.join(output_subfolder, f"mask_{filename}")
                cv2.imwrite(output_path, mask_output)
                print(f"Masque généré : {output_path}")
