import os
import cv2
import numpy as np


#seed = 2
#threshold = 0.9

def convert_red_to_binary_mask(input_path, output_path):
    # Liste des catégories
    categories = ["chemical_contamination", "dust", "good", "scratch"]

    for category in categories:
        input_dir = os.path.join(input_path, category)
        output_dir = os.path.join(output_path, category)
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_image_path = os.path.join(input_dir, filename)
                output_image_path = os.path.join(output_dir, filename)

                # Lire l'image couleur
                image = cv2.imread(input_image_path)

                # Convertir en HSV pour détecter le rouge
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                # Masque pour le rouge (deux plages en HSV)
                lower_red1 = np.array([0, 70, 50])
                upper_red1 = np.array([10, 255, 255])
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

                lower_red2 = np.array([160, 70, 50])
                upper_red2 = np.array([180, 255, 255])
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

                # Union des deux masques rouges
                red_mask = cv2.bitwise_or(mask1, mask2)

                # Tout ce qui est rouge devient blanc (255), le reste reste noir (0)
                binary_mask = cv2.threshold(red_mask, 1, 255, cv2.THRESH_BINARY)[1]

                # Sauvegarder le masque binaire
                cv2.imwrite(output_image_path, binary_mask)

    print(f"all is done !")
    
shot = 16
    
S = [0, 1, 2]
T = [0.2, 0.3, 0.4]
K = [9, 10, 11]
    
for seed in S : 
    for threshold in T :
        # Exemple d'utilisation
        for k_neighbors in K :
            
            input_path = f"/feynman/work/dedip/lemid/oe283118/AnomalyDINO/results/dataset_original_size_256_AnomalyDINO{shot}_seed{seed}_threshold{threshold}_k_neighbors{k_neighbors}"
            output_path = f"/feynman/work/dedip/lemid/oe283118/AnomalyDINO/results/dataset_original_size_256_AnomalyDINO{shot}_seed{seed}_threshold{threshold}_k_neighbors{k_neighbors}_output_mask"
            convert_red_to_binary_mask(input_path, output_path)
