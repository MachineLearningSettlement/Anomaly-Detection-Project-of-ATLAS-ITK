import os


ground_truth_path = "/feynman/home/dedip/lemid/oe283118/work/anomalib/datasets/MVTecAD/maindata_resize128_PATCHCORE/ground_truth"

# Parcours des sous-dossiers
for defect_type in os.listdir(ground_truth_path):
    defect_path = os.path.join(ground_truth_path, defect_type)
    
    # Vérifie que c'est bien un dossier
    if os.path.isdir(defect_path):
        for filename in os.listdir(defect_path):
            #if filename.endswith(".png") and not filename.endswith("_mask.png"):
            if  filename.endswith("_mask.png"):

                old_path = os.path.join(defect_path, filename)
                
                #name, ext = os.path.splitext(filename)
                #new_filename = f"{name}_mask{ext}"
                new_filename = filename.replace("_mask", "")   # <== correction ici
                new_path = os.path.join(defect_path, new_filename)
                
                os.rename(old_path, new_path)
                print(f"Renommé : {filename} -> {new_filename}")
