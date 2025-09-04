import os
import cv2
import numpy as np
from sklearn.metrics import f1_score

def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (mask > 0).astype(np.uint8)

def get_all_image_paths(root_dir):
    all_paths = []
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.endswith((".png", ".jpg", ".jpeg")):
                    all_paths.append((class_name, os.path.join(class_dir, file)))
    return sorted(all_paths)

def evaluate_metrics(gt_dir, pred_dir):
    pixel_gt, pixel_pred = [], []
    global_gt, global_pred = [], []

    gt_paths = get_all_image_paths(gt_dir)
    pred_paths = get_all_image_paths(pred_dir)

    assert len(gt_paths) == len(pred_paths), "Les deux dossiers doivent contenir le même nombre d'images"

    for (gt_class, gt_path), (pred_class, pred_path) in zip(gt_paths, pred_paths):
        assert gt_class == pred_class, f"Classe différente: {gt_class} vs {pred_class}"
        # Si tu compares strictement par ordre et pas par nom, tu peux commenter la ligne suivante
        assert os.path.basename(gt_path) == os.path.basename(pred_path), f"Noms différents : {gt_path} vs {pred_path}"

        gt_mask = load_mask(gt_path)
        pred_mask = load_mask(pred_path)

        pixel_gt.extend(gt_mask.flatten())
        pixel_pred.extend(pred_mask.flatten())

        global_gt.append(int(gt_mask.sum() > 0))
        global_pred.append(int(pred_mask.sum() > 0))
        
    

    f1_pixel = f1_score(pixel_gt, pixel_pred)
    f1_global = f1_score(global_gt, global_pred)

    return {
        "F1 Pixel": f1_pixel,
        "F1 Global": f1_global
    }

if __name__ == "__main__":
    
    shot = 16
    seed = 0
    threshold = 0.2
    k_neighbors = 9
    
    S = [0, 1, 2]
    T = [0.2, 0.3, 0.4]
    K = [9, 10, 11, 12]
    
    #for seed in S :
        #for threshold in T : 
           # for k_neighbors in K :
                
ground_truth_dir = "/feynman/work/dedip/lemid/oe283118/AnomalyDINO/results/dataset_original_size_256_AnomalyDINO_masks_gth"
prediction_dir = f"/feynman/work/dedip/lemid/oe283118/AnomalyDINO/results/dataset_original_size_256_AnomalyDINO16_seed0_threshold0.2_k_neighbors10_output_mask"

results = evaluate_metrics(ground_truth_dir, prediction_dir)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
                #print(f"la ligne de {seed}_{threshold}_{k_neighbors}")
                #print("__________________")
            


        
        