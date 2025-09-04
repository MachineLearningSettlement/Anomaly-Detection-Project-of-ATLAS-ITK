import os
import torch
import gc
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from sklearn.model_selection import ParameterGrid
from anomalib.metrics import AUROC
from anomalib.metrics import F1Score
from safetensors.torch import load_file
from torchvision.models import wide_resnet50_2
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os



#  Optionnel : meilleure gestion mémoire CUDA (facultatif mais conseillé)
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

param_grid = {'num_neighbors' : [14] }

dict_result = {
    'configuration' : [] ,
    'numéro de configuration' : [] ,
    'image_auroc' : [] ,
    'pixel_auroc' : [] ,
    'image_f1' : [] ,
    'pixel_f1' : []
}

model = wide_resnet50_2(pretrained=False) 
model.name = 'wide_resnet50_2'

model_path = 'model.safetensors'  

state_dict = load_file(model_path)

# Charger les poids dans le modèle
model.load_state_dict(state_dict)



i = 1
for params in ParameterGrid(param_grid): 
    
    
    print("configuration :", params)
    
    print("numéro de configuration :", i)
    
    #print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    #print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    #print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    # Création des objets
    engine = Engine()
    

    datamodule = MVTecAD(
        root="/feynman/work/dedip/lemid/oe283118/anomalib/datasets/MVTecAD",
        category="dataset_original_size_256_threshlold_50",
    )
    
    patchcore_model = Patchcore(
    backbone=model,  
    layers=['layer1', 'layer2'],
    num_neighbors=params['num_neighbors'],
    coreset_sampling_ratio=0.01 # 32% max pour new old in feynman 04
    )
    
    # Entraînement
    engine.fit(model=patchcore_model, datamodule=datamodule)

    # Prédictions
    image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    image_f1 = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
    pixel_f1 = F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_")
    
    predictions = engine.predict(model=patchcore_model, datamodule=datamodule)
    
    
    for batch in predictions:
        image_auroc.update(batch)
        pixel_auroc.update(batch)
        image_f1.update(batch)
        pixel_f1.update(batch)
    
    
    
    print(image_auroc.name, image_auroc.compute())  
    print(pixel_auroc.name, pixel_auroc.compute())
    print(image_f1.name, image_f1.compute())  
    print(pixel_f1.name, pixel_f1.compute())
    
    dict_result['configuration'].append(params)
    dict_result['numéro de configuration'].append(i)
    dict_result['image_auroc'].append(image_auroc.compute())
    dict_result['pixel_auroc'].append(pixel_auroc.compute())
    dict_result['image_f1' ].append(image_f1.compute())
    dict_result['pixel_f1'].append(pixel_f1.compute())
    
    #réinitialiser le compteur 
    i += 1

    # Nettoyage mémoire GPU
    del predictions
    del patchcore_model
    del datamodule
    del engine
    gc.collect()

    #print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    #print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    #print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

     
    
    print("________________________________________________________")
    

print(dict_result)