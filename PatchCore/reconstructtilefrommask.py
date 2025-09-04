import os
import cv2
import numpy as np

def load_image(path):
    return cv2.imread(path)

def load_mask(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def overlay_mask_on_image(image, mask):
    """Projette le masque sur l'image avec la couleur rouge (255, 0, 0 en BGR)."""
    result = image.copy()
    red = np.array([0, 0, 255], dtype=np.uint8)  # OpenCV: BGR
    result[mask == 255] = red
    return result

def get_all_image_paths(root_dir):
    all_paths = []
    for class_name in os.listdir(root_dir):
        if class_name == 'good':
            continue  # Ignore le dossier "good"
        
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    all_paths.append((class_name, os.path.join(class_dir, file)))
    
    return sorted(all_paths)

def superpose_masks(originals_dir, masks_dir, output_dir):
    original_paths = get_all_image_paths(originals_dir)
    mask_paths = get_all_image_paths(masks_dir)

    assert len(original_paths) == len(mask_paths), "Les deux dossiers doivent contenir le même nombre d'images"

    for (orig_class, orig_path), (mask_class, mask_path) in zip(original_paths, mask_paths):
        assert orig_class == mask_class, f"Classe différente: {orig_class} vs {mask_class}"
        assert os.path.basename(orig_path) == os.path.basename(mask_path), f"Noms différents : {orig_path} vs {mask_path}"

        image = load_image(orig_path)
        mask = load_mask(mask_path)

        assert image.shape[:2] == mask.shape[:2], f"Dimensions différentes : {orig_path}"

        image_overlay = overlay_mask_on_image(image, mask)

        output_class_dir = os.path.join(output_dir, orig_class)
        os.makedirs(output_class_dir, exist_ok=True)

        output_path = os.path.join(output_class_dir, os.path.basename(orig_path))
        cv2.imwrite(output_path, image_overlay)

    print("[✔] Superposition terminée. Résultats enregistrés dans :", output_dir)

if __name__ == "__main__":
    originals_dir = "/feynman/home/dedip/lemid/oe283118/work/anomalib/datasets/MVTecAD/itkPolar_ParisPreProd_128_clean_brut_resize2_70/test"
    masks_dir = "/feynman/home/dedip/lemid/oe283118/work/results/Patchcore/itkPolar_ParisPreProd_128_clean_brut_resize2_70_copie_output_mask_intersection"
    output_dir = "/feynman/home/dedip/lemid/oe283118/work/results/Patchcore/itkPolar_ParisPreProd_128_clean_brut_resize2_70_reconstructing_tiles"

    superpose_masks(originals_dir, masks_dir, output_dir)
