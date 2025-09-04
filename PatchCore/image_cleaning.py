from os import listdir, makedirs
from os.path import isfile, join

import cv2
import numpy as np

from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()

    # dataset path:
    parser.add_argument('--dir_input_clean', type=str, default="/local/home/oe283118/Bureau/AtlasITKDataSets/dataset_size_128/original_pics/",
    help="Directory with original images")
    parser.add_argument('--dir_output_clean', type=str, default="/local/home/oe283118/Bureau/AtlasITKDataSets/dataset_size_128/cleaned_images/",
    help="Directory with cleaned images")
    parser.add_argument('--dir_masks', type=str, default="/local/home/oe283118/Bureau/AtlasITKDataSets/dataset_size_128/masks/",
    help="Directory with masks")
    parser.add_argument('--dir_input_dataset', type=str, default="/local/home/oe283118/Bureau/AtlasITKDataSets/dataset_size_128/cleaned_images/",
    help="Directory with input images for dataset")
    parser.add_argument('--dir_output_dataset', type=str, default="/local/home/oe283118/Bureau/AtlasITKDataSets/dataset_size_128/main_dataset_size_128/",
    help="Directory with generated dataset")

    parser.add_argument('--need_cleaning', type=bool, default=False, help="True if need to create images without background")

    parser.add_argument('--tile_size', type=int, default=128, help="Tile size in pixels") #256x2=512
    parser.add_argument('--train_part', type=float, default=0.7, help="Part of good images to train")

    args = parser.parse_args()
    return args

def apply_color_transformation(img, target_mean, target_std, mask = None):
    # Convert the image to float32
    img = img.astype('float32')
    # Calculate the current mean and standard deviation of each channel
    current_mean, current_std = cv2.meanStdDev(img, mask = mask)
    # Apply the color transformation
    for i in range(3):  # For each channel (L, A, and B)
        img[:,:,i] = (img[:,:,i] - current_mean[i]) * (target_std[i] / current_std[i]) + target_mean[i]
    # Clip the values to be in the valid range (0 to 255)
    img = np.clip(img, 0, 255).astype('uint8')

    return img


def deleteBck(image):
    ########################################################
    ## Gets openCV image of a module                      ##
    ## Outputs same image with everything but pads masked ##
    ########################################################

    # 1.1 Apply Otsu thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (35,35), 0)
    _, thresholdMask = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    cnorm = {'mu': np.array([196.74390367, 124.80378614, 143.89042235]),
             'sigma': np.array([33.18296434,1.54909409 , 2.84252947]),
            }


    # Convert PIL.Image to numpy array
    image = np.array(image)

    # 1.1 Apply Otsu thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresholdMask = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 1.2 Adjust obtained mask
    dilate = cv2.dilate(thresholdMask, np.ones((20, 20)))
    mask_otsu  = cv2.erode(dilate, np.ones((21, 21)))

    # Apply per colour channel normalization
    renormed_result = apply_color_transformation(cv2.cvtColor(image, cv2.COLOR_BGR2LAB),cnorm['mu'],cnorm['sigma'], mask = mask_otsu)
    image = cv2.cvtColor(renormed_result, cv2.COLOR_LAB2BGR)

    image_cleaned = cv2.bitwise_and(image, image, mask=mask_otsu)


    return image_cleaned

def transform_to_white(curr_mask):

    red = np.array([255, 0, 0])    
    green = np.array([0, 255, 0])  
    blue = np.array([0, 0, 255])   
    
    white = np.array([255, 255, 255])  
    
    for i in range(curr_mask.shape[0]):  
        for j in range(curr_mask.shape[1]):  
            if np.array_equal(curr_mask[i, j], red) or np.array_equal(curr_mask[i, j], green) or np.array_equal(curr_mask[i, j], blue):
                curr_mask[i, j] = white

    return curr_mask

if __name__ == '__main__':
    args = parse_arguments() 
    print("Arguments are parsed")

    # Clean if need cleaning
    if args.need_cleaning:
        image_clean_filenames = [f for f in listdir(args.dir_input_clean) if isfile(join(args.dir_input_clean, f))]
        print(image_clean_filenames)
        for filename in image_clean_filenames:
            image = cv2.imread(args.dir_input_clean + filename)
            print(filename, image.shape)
            image_cleaned = deleteBck(image)
            cv2.imwrite(args.dir_output_clean + filename, image_cleaned)
        print("Deleting background: done")

    # Split into tiles if there is a mask for this image
    masks_filenames = [f for f in listdir(args.dir_masks) if isfile(join(args.dir_masks, f))]# and "Flex" in join(args.dir_input_clean, f)]
    all_images_filenames = [f for f in listdir(args.dir_input_dataset) if isfile(join(args.dir_input_dataset, f))]# and "Flex" in join(args.dir_input_clean, f)]

    wTile, hTile = (args.tile_size, args.tile_size)

    print(masks_filenames)
    print(all_images_filenames)
    
    # All this because of different extentions

    images_filenames = []

    for mask_filename in masks_filenames:
        mask_name = mask_filename.split(".")[0]
        same = [string for string in all_images_filenames if mask_name in string]
        print(same)
        if same:
            images_filenames.append(same[0])
            
        print(images_filenames)
    
    for i in range(len(images_filenames)):
        mask_filename = masks_filenames[i]
        image_filename = images_filenames[i]

        mask_name, mask_ext   = mask_filename.split(".")
        image_name, image_ext = image_filename.split(".")

        print(mask_name, mask_ext)
        print(image_name, image_ext)

        # Some image; get width and height
        image = cv2.imread(join(args.dir_input_dataset, image_filename))
        mask  = cv2.imread(join(args.dir_masks, mask_filename))
     
        mask[mask >= 128] = 255
        mask[mask <  128] = 0
        mask[np.where((mask==[255, 0, 255]).all(axis=2))] = [255, 0, 0]

        h, w = image.shape[:2]

        # Number of tiles
        nTilesX = np.uint8(np.ceil(w / wTile))
        nTilesY = np.uint8(np.ceil(h / hTile))

        image = cv2.copyMakeBorder(image, 0, nTilesY*hTile-h, 0, nTilesX*wTile-w, cv2.BORDER_CONSTANT, value=0)
        mask  = cv2.copyMakeBorder(mask,  0, nTilesY*hTile-h, 0, nTilesX*wTile-w, cv2.BORDER_CONSTANT, value=0)

        for i in range(nTilesX):
            for j in range(nTilesY):
                curr_img  = image[j*hTile:(j+1)*hTile, i*wTile:(i+1)*wTile]
                curr_mask = mask[ j*hTile:(j+1)*hTile, i*wTile:(i+1)*wTile]

                image_out = f'{image_name}_{i*wTile}_{j*hTile}{"." + mask_ext}'
                mask_out  = f'{image_name}_{i*wTile}_{j*hTile}{"_mask." + mask_ext}'

                if np.sum(curr_mask) == 0:
                    if np.random.rand(1) > args.train_part:
                        cv2.imwrite(args.dir_output_dataset + "test/good/" + image_out, curr_img)
                    else:
                        cv2.imwrite(args.dir_output_dataset + "train/good/" + image_out, curr_img)
                else:
                    if 255 in curr_mask[:,:,2]:
                        cv2.imwrite(args.dir_output_dataset + "test/chemical_contamination/" + image_out, curr_img)
                        cv2.imwrite(args.dir_output_dataset + "ground_truth/chemical_contamination/" + mask_out, transform_to_white(curr_mask))
                    elif 255 in curr_mask[:,:,1]:
                        cv2.imwrite(args.dir_output_dataset + "test/dust/" + image_out, curr_img)
                        cv2.imwrite(args.dir_output_dataset + "ground_truth/dust/" + mask_out, transform_to_white(curr_mask))
                    elif 255 in curr_mask[:,:,0]:
                        cv2.imwrite(args.dir_output_dataset + "test/scratch/" + image_out, curr_img)
                        cv2.imwrite(args.dir_output_dataset + "ground_truth/scratch/" + mask_out, transform_to_white(curr_mask))
                    
        print("Done with ", image_filename)

        """# Total remainders
        remainderX = nTilesX * wTile - w
        remainderY = nTilesY * hTile - h

        # Set up remainders per tile
        remaindersX = np.ones((nTilesX-1, 1)) * np.uint16(np.floor(remainderX / (nTilesX-1)))
        remaindersY = np.ones((nTilesY-1, 1)) * np.uint16(np.floor(remainderY / (nTilesY-1)))
        remaindersX[0:np.remainder(remainderX, np.uint16(nTilesX-1))] += 1
        remaindersY[0:np.remainder(remainderY, np.uint16(nTilesY-1))] += 1

        # Initialize array of tile boxes
        tiles = np.zeros((nTilesX * nTilesY, 4), np.uint16)
 
        print("img size: ", h, w)
        print("tile size: ", hTile, wTile)
        print("rem: ", remaindersX, remaindersY)
        print(tiles.shape)
       

        # Determine proper tile boxes
        k = 0
        x = 0
        for i in range(nTilesX):
            y = 0
            for j in range(nTilesY):
                tiles[k, :] = (x, y, hTile, wTile)
                k += 1
                if (j < (nTilesY-1)):
                    y = y + hTile - remaindersY[j]
            if (i < (nTilesX-1)):
                x = x + wTile - remaindersX[i]

        for i, tile_shape in enumerate(tiles):
            x, y, hx, hy = tile_shape
            curr_img = image[y:y+hy, x:x+hx]
            curr_mask = mask[y:y+hy, x:x+hx]

            image_out = f'{image_name}_{x}_{y}{"." + mask_ext}'
            mask_out  = f'{image_name}_{x}_{y}{"_mask." + mask_ext}'

            if np.sum(curr_mask) == 0:
                if np.random.rand(1) > args.train_part:
                    cv2.imwrite(args.dir_output_dataset + "test/good/" + image_out, curr_img)
                else:
                    cv2.imwrite(args.dir_output_dataset + "train/good/" + image_out, curr_img)
            else:
                if 255 in curr_mask[:,:,2]:
                    cv2.imwrite(args.dir_output_dataset + "test/chemical_contamination/" + image_out, curr_img)
                    cv2.imwrite(args.dir_output_dataset + "ground_truth/chemical_contamination/" + mask_out, curr_mask[:,:,2])
                if 255 in curr_mask[:,:,1]:
                    cv2.imwrite(args.dir_output_dataset + "test/dust/" + image_out, curr_img)
                    cv2.imwrite(args.dir_output_dataset + "ground_truth/dust/" + mask_out, curr_mask[:,:,1])
                if 255 in curr_mask[:,:,0]:
                    cv2.imwrite(args.dir_output_dataset + "test/scratch/" + image_out, curr_img)
                    cv2.imwrite(args.dir_output_dataset + "ground_truth/scratch/" + mask_out, curr_mask[:,:,0])"""