import os

dossier1 = "/feynman/work/dedip/lemid/oe283118/AnomalyDINO/results/dataset_original_size_256_AnomalyDINO_masks_gth"
dossier2 = "/feynman/work/dedip/lemid/oe283118/AnomalyDINO/results/dataset_original_size_256_AnomalyDINO16_seed0_threshold0.2_k_neighbors9_output_mask"
l1, l2, l3 = [],[], []

for file1 in os.listdir(dossier1):
    #print(file1)   
    l1.append(file1)
#print("-----------------------")
for file2 in os.listdir(dossier2):
    #print(file2)  
    l2.append(file2)

        
        
print(len(l1))
print(len(l2))
#print(len(l3))


print('likain f test o makaynch f gth')
for var in l2 :
    
    if var not in l1 :
        
        print(var)
print('-------------------------------------------')        

print('likain f gth o makaynch f test') 
for var in l1 :
    
    if var not in l2 :
        
        print(var)