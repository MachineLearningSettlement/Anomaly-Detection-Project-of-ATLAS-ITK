import os
import cv2
import numpy as np

def delete_black_images(folder_path):
    i = 0
    for filename in os.listdir(folder_path):
        i = i+1
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            if image is None:
                continue
            if np.all(image == 0):  # 100% black
                os.remove(file_path)
                print(f"Image : {i} is Done ")

if __name__ == '__main__':
    
    folder_path = "/local/home/oe283118/Bureau/AtlasITKDataSets/dataset_size_128/main_dataset_size_128/train/good"
    folder_path1 = "/local/home/oe283118/Bureau/AtlasITKDataSets/dataset_size_128/main_dataset_size_128/test/good"
    delete_black_images(folder_path1)
