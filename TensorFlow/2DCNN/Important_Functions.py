from __future__ import division
import os
import cv2
import h5py
import random
import shutil
import numpy as np
from sys import exit
from PIL import Image
from tqdm import tqdm
from scipy import interp
import albumentations as A
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import binary_erosion, binary_fill_holes


# To select the same images
# random.seed(1)

def process_raw_data(data_path, process_list):
    Class_Names = os.listdir(f'{data_path}')
    transform = A.Compose(process_list)
    for ii in tqdm(range(0, len(Class_Names))):
        # print(f'Current Class: {Class_Names[ii]}')  # Print Current Class Name
        # List containing all images of a certain class in the Raw Dataset
        Image_List = os.listdir(f'{data_path}/{Class_Names[ii]}')
        for iii in range(0, len(Image_List)):
            current_image = os.path.splitext(Image_List[iii])[0]
            # Read an image with OpenCV and process
            org_image = cv2.imread(f'{data_path}/{Class_Names[ii]}/{Image_List[iii]}')  # Read Original Image
            img_nparray = np.asarray(org_image)
            if img_nparray.shape[2] == 4:
                org_image = org_image[:,:,:3]  # Remove Alpha (4th) Channel from the Image if required
            org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB Colorspace, cv2 by default reads in BGR format
            transformed = transform(image=org_image)  # Augment Image
            transformed_image = transformed["image"]
            # transformed_image = cv2.resize(transformed_image, (331, 331), interpolation=cv2.INTER_CUBIC)  # Resize Image
            cv2.imwrite(f'{data_path}/{Class_Names[ii]}/{current_image}.png', transformed_image)  # Replacing the Original Image with a Transformed Version


def copyimagefile(source_path, destination_path, files_list):
    for order in range(1, len(files_list)):
        files = files_list[order]
        shutil.copyfile(os.path.join(source_path, files), os.path.join(destination_path, files))


def create_folds(raw_data_path, num_folds, train_portion, validation_portion=False):
  random.seed(1)
  Class_Names = os.listdir(raw_data_path)
  for i in range(1, num_folds + 1):
      print(f'Creating Fold {i}')
      for ii in tqdm(range(0, len(Class_Names))):
          # Get Train and Test Image Indices Randomly for each Fold
          source_path = f'{raw_data_path}/{Class_Names[ii]}'
          if (ii == 0):
            X_Tot = os.listdir(source_path)  # List containing all images
            X_Train_Len = int(len(X_Tot) * train_portion)
            X_Train = random.sample(X_Tot, (X_Train_Len - 1))  # Randomly formed List containing images for training, can be stratified otherwise
            X_Test = [x for x in X_Tot if x not in X_Train]  # List containing images for testing
            X_Val = []
            if validation_portion:
                X_Val_Len = int(len(X_Train) * validation_portion)  # Size of the Validation Set, subset of the Training Set
                X_Val = random.sample(X_Train, (X_Val_Len - 1))  # Validation Set
                X_Train = [x for x in X_Train if x not in X_Val]  # Validation Set is deducted from the Training Set

          # Make Required Directories after Checking their Existence, sometimes delete old folders and run the code again
          train_dir = f'Data/Train/fold_{i}/{Class_Names[ii]}'
          test_dir = f'Data/Test/fold_{i}/{Class_Names[ii]}'
          val_dir = f'Data/Val/fold_{i}/{Class_Names[ii]}'
          if not os.path.isdir(train_dir):
              os.makedirs(train_dir)  # Train Directory for Fold ii
          if not os.path.isdir(test_dir):
              os.makedirs(test_dir)  # Test Directory for Fold ii
          if (not os.path.isdir(val_dir)) and (validation_portion != False):
              os.makedirs(val_dir)  # Validation Directory for Fold ii

          # Copy Image Files from the Source Folder to the Destination Folder
          copyimagefile(source_path, train_dir, X_Train)
          copyimagefile(source_path, test_dir, X_Test)
          if validation_portion:
              copyimagefile(source_path, val_dir, X_Val)  # True if validation set is created independently


def augment(data_path, num_folds, augmentation_list, augmentation_num):
    Class_Names = os.listdir(f'{data_path}/fold_1')
    # Declare an Augmentation Pipeline
    transform = A.Compose(augmentation_list)
    for i in range(1, num_folds + 1):
        print(f'\nCurrently Processing Fold {i}')
        for ii in tqdm(range(0, len(Class_Names))):
            # print(f'Current Class: {Class_Names[ii]}')
            # List containing all images of a certain class in a certain fold
            X_Train_List = os.listdir(f'{data_path}/fold_{i}/{Class_Names[ii]}')
            for iii in range(0, len(X_Train_List)):
                current_image = os.path.splitext(X_Train_List[iii])[0]
                # Read an image with OpenCV and convert it to the RGB colorspace
                org_image = cv2.imread(f'{data_path}/fold_{i}/{Class_Names[ii]}/{X_Train_List[iii]}')  # Read Original Image
                img_nparray = np.asarray(org_image)
                if img_nparray.shape[2] == 4:
                    org_image = org_image[:,:,:3]  # Remove Alpha (4th) Channel from the Image
                org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB Colorspace
                for iv in range(1, augmentation_num + 1):
                    transformed = transform(image=org_image)  # Augment Image
                    transformed_image = transformed["image"]
                    cv2.imwrite(f'{data_path}/fold_{i}/{Class_Names[ii]}/{current_image}_Augmented_{iv}.png', transformed_image)


def get_datasets(imgs_dir, groundTruth_dir, height, width, channels, train_test="null"):
    Nimgs = len(os.listdir(imgs_dir))  # List containing all images
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir): # List all files, directories in the path
        for i in range(len(files)):
            # Original
            # print("Original image: "+files[i])
            img = Image.open(imgs_dir+'/'+files[i])
            imgs[i] = np.asarray(np.expand_dims(img, axis=2))
            # Corresponding Ground Truth
            groundTruth_name = files[i]
            # print ("Ground Truth Name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir+'/'+groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)

    print ("Max Pixel Value for Images: " +str(np.max(imgs)))
    print ("Max Pixel Value for Masks: " +str(np.min(imgs)))
    assert(np.max(groundTruth)==255)
    assert(np.min(groundTruth)==0)
    print ("Ground Truth Masks are correctly withih pixel value range 0-255 (black-white)")
    # Reshaping for my standard tensors
    # imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,height,width,channels))
    groundTruth = np.reshape(groundTruth,(Nimgs,height,width,1))
    assert(groundTruth.shape == (Nimgs,height,width,1))

    return imgs, groundTruth


def load_hdf5(infile):
  with h5py.File(infile,"r") as f:  # "with" close the file after its nested commands
    return f["images"][()]


def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("images", data=arr, dtype=arr.dtype)


# Convert RGB image in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  # 4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs


# Dice Similarity Index
def dice(true, pred, k=1):
    intersection = np.sum(pred[true == k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))

    return dice


# Group a set of images row per columns
def group_images(data,per_row):
    assert data.shape[0]%per_row==0
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))  #corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg


# Visualize image (as PIL image, NOT as matplotlib!)
def visualize(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    img = None
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')
    return img


# Prepare the mask in the right shape for the Unet
def masks_Unet(masks):
    assert (len(masks.shape)==4)  #4D arrays
    assert (masks.shape[1]==1 )  #check the channel is 1
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks,(masks.shape[0],im_h*im_w))
    new_masks = np.empty((masks.shape[0],im_h*im_w,2))
    for i in range(masks.shape[0]):
        for j in range(im_h*im_w):
            if  masks[i,j] == 0:
                new_masks[i,j,0]=1
                new_masks[i,j,1]=0
            else:
                new_masks[i,j,0]=0
                new_masks[i,j,1]=1
    return new_masks


def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    assert (pred.shape[2]==2 )  #check the classes are 2
    pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
    if mode=="original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i,pix]=pred[i,pix,1]
    elif mode=="threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print ("Mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images,(pred_images.shape[0],1, patch_height, patch_width))
    return pred_images


# My pre processing (use for both training and testing!)
def my_PreProc(data):
    # assert(len(data.shape)==4)
    if len(data.shape) == 3:
      data = np.expand_dims(data, axis=3) 
    # black-white conversion
    if data.shape[3] == 3:
      data = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(data)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    
    return train_imgs


# PRE PROCESSING FUNCTIONS
# Histogram Equalization
def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[3]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# Adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[3]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# Normalize over the dataset
def dataset_normalized(imgs):
    assert(len(imgs.shape)==4)
    assert(imgs.shape[3]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[3]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs


# Load the original data and return the extracted patches for training/testing
def get_data_training(Train_Images, Train_Masks, patch_height, patch_width, N_subimgs, inside_FOV):
    input_type = str(isinstance(Train_Images, str))
    if input_type == True:
      train_imgs_original = load_hdf5(Train_Images)
      train_masks = load_hdf5(Train_Masks)
    else:
      train_imgs_original = Train_Images
      train_masks = Train_Masks
    # visualize(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train')#.show()  #check original imgs train

    train_imgs = my_PreProc(train_imgs_original)
    if len(train_masks.shape) == 3:
      train_masks = np.expand_dims(train_masks, axis=3) 
    train_masks = train_masks/255.

    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    # assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print ("\nTrain images/masks shape:")
    print (train_imgs.shape)
    print ("Train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print ("Train masks are within 0-1\n")

    #extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train = extract_random(train_imgs, train_masks, patch_height, patch_width, N_subimgs, inside_FOV)
    data_consistency_check(patches_imgs_train, patches_masks_train)

    print ("\nTrain patch images/masks shape:")
    print (patches_imgs_train.shape)
    print ("Train patch images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train#, patches_imgs_test, patches_masks_test


# Load the original data and return the extracted patches for training/testing
def get_data_testing(Test_Image_Set, Test_Mask_Set, Imgs_to_test, patch_height, patch_width):
    ### test
    input_type = str(isinstance(Test_Image_Set, str))
    if input_type == True:
      test_imgs_original = load_hdf5(Test_Image_Set)
      test_masks = load_hdf5(Test_Mask_Set)
    else:
      test_imgs_original = Test_Image_Set
      test_masks = Test_Mask_Set

    test_imgs = my_PreProc(test_imgs_original)
    test_masks = test_masks/255.

    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
    test_masks = test_masks[0:Imgs_to_test,:,:,:]
    test_imgs = paint_border(test_imgs,patch_height,patch_width)
    test_masks = paint_border(test_masks,patch_height,patch_width)

    data_consistency_check(test_imgs, test_masks)

    #check masks are within 0-1
    assert(np.max(test_masks)==1  and np.min(test_masks)==0)

    print ("\ntest images/masks shape:")
    print (test_imgs.shape)
    print ("Test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print ("Test masks are within 0-1\n")

    #extract the TEST patches from the full images
    patches_imgs_test = extract_ordered(test_imgs,patch_height,patch_width)
    patches_masks_test = extract_ordered(test_masks,patch_height,patch_width)
    data_consistency_check(patches_imgs_test, patches_masks_test)

    print ("\nTest patch images/masks shape:")
    print (patches_imgs_test.shape)
    print ("Test patch images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))

    return patches_imgs_test, patches_masks_test


# Load the original data and return the extracted patches for testing
# Return the ground truth in its original shape
def get_data_testing_overlap(Test_Image_Set, Test_Mask_Set, Imgs_to_test, patch_height, patch_width, stride_height, stride_width):
    ### test
    input_type = isinstance(Test_Image_Set, str)
    if input_type == True:
      test_imgs_original = load_hdf5(Test_Image_Set)
      test_imgs_original = np.einsum('klij->kijl', test_imgs_original)
      test_masks = load_hdf5(Test_Mask_Set)
      test_masks = np.einsum('klij->kijl', test_masks)
    else:
      test_imgs_original = Test_Image_Set
      test_masks = Test_Mask_Set
    
    if len(test_imgs_original.shape) == 3:
      test_imgs_original = np.expand_dims(test_imgs_original, axis=3)
    if len(test_masks.shape) == 3:
      test_masks = np.expand_dims(test_masks, axis=3)
    test_imgs = my_PreProc(test_imgs_original)
    test_masks = test_masks/255.
    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
    test_masks = test_masks[0:Imgs_to_test,:,:,:]
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)

    # check masks are within 0-1
    # assert(np.max(test_masks)==1  and np.min(test_masks)==0)

    print ("\nTest images shape:")
    print (test_imgs.shape)
    print ("\nTest mask shape:")
    print (test_masks.shape)
    print ("Test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print ("Test masks are within 0-1\n")

    #extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap(test_imgs,patch_height,patch_width,stride_height,stride_width)

    print ("\nTest patch images shape:")
    print (patches_imgs_test.shape)
    print ("Test patch images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))

    return patches_imgs_test, test_imgs.shape[1], test_imgs.shape[2], test_masks


# Data consinstency check
def data_consistency_check(imgs,masks):
    assert(len(imgs.shape)==len(masks.shape))
    assert(imgs.shape[0]==masks.shape[0])
    assert(imgs.shape[1]==masks.shape[1])
    assert(imgs.shape[2]==masks.shape[2])
    assert(masks.shape[3]==1)
    assert(imgs.shape[3]==1 or imgs.shape[3]==3)


# Extract patches randomly in the full training images, Inside OR in full image
def extract_random(full_imgs, full_masks, patch_h, patch_w, N_patches, inside=True):
    if (N_patches%full_imgs.shape[0] != 0):
        print ("N_patches: plase enter a multiple of 20")
        exit()
    assert (len(full_imgs.shape)==4 and len(full_masks.shape)==4)  #4D arrays
    assert (full_imgs.shape[3]==1 or full_imgs.shape[3]==3)  #check the channel is 1 or 3
    assert (full_masks.shape[3]==1)   #masks only black and white
    assert (full_imgs.shape[1] == full_masks.shape[1] and full_imgs.shape[2] == full_masks.shape[2])
    patches = np.empty((N_patches,patch_h,patch_w,full_imgs.shape[3]))
    patches_masks = np.empty((N_patches,patch_h,patch_w,full_masks.shape[3]))
    img_h = full_imgs.shape[1]  #height of the full image
    img_w = full_imgs.shape[2] #width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches/full_imgs.shape[0])  #N_patches equally divided in the full images
    print ("Patches per full image: " +str(patch_per_img))
    iter_tot = 0   #iter over the total numbe rof patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        k=0
        while k < patch_per_img:
            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
            # print "x_center " +str(x_center)
            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
            # print "y_center " +str(y_center)
            #check whether the patch is fully contained in the FOV
            if inside==True:
                if is_patch_inside_FOV(x_center,y_center,img_w,img_h,patch_h)==False:
                    continue
            patch = full_imgs[i,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2),:]
            patch_mask = full_masks[i,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2),:]
            patches[iter_tot]=patch
            patches_masks[iter_tot]=patch_mask
            iter_tot +=1   #total
            k+=1  #per full_img
    
    return patches, patches_masks


# Check if the patch is fully contained in the FOV
def is_patch_inside_FOV(x,y,img_w,img_h,patch_h):
    x_ = x - int(img_w/2) # origin (0,0) shifted to image center
    y_ = y - int(img_h/2)  # origin (0,0) shifted to image center
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0) #radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
    radius = np.sqrt((x_*x_)+(y_*y_))
    if radius < R_inside:
        return True
    else:
        return False


# Divide all the full_imgs in pacthes
def extract_ordered(full_imgs, patch_h, patch_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    N_patches_h = int(img_h/patch_h) #round to lowest int
    if (img_h%patch_h != 0):
        print ("Warning: " +str(N_patches_h) +" patches in height, with about " +str(img_h%patch_h) +" pixels left over")
    N_patches_w = int(img_w/patch_w) #round to lowest int
    if (img_h%patch_h != 0):
        print ("Warning: " +str(N_patches_w) +" patches in width, with about " +str(img_w%patch_w) +" pixels left over")
    print ("Number of patches per image: " +str(N_patches_h*N_patches_w))
    N_patches_tot = (N_patches_h*N_patches_w)*full_imgs.shape[0]
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))

    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = full_imgs[i,:,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches  #array with all the full_imgs divided in patches


def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[3]==1 or full_imgs.shape[3]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[1]  #height of the full image
    img_w = full_imgs.shape[2] #width of the full image
    leftover_h = (img_h-patch_h)%stride_h  #leftover on the h dim
    leftover_w = (img_w-patch_w)%stride_w  #leftover on the w dim
    if (leftover_h != 0):  #change dimension of img_h
        print ("\nthe side H is not compatible with the selected stride of " +str(stride_h))
        print ("img_h " +str(img_h) + ", patch_h " +str(patch_h) + ", stride_h " +str(stride_h))
        print ("(img_h - patch_h) MOD stride_h: " +str(leftover_h))
        print ("So the H dim will be padded with additional " +str(stride_h - leftover_h) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0],img_h+(stride_h-leftover_h),img_w,full_imgs.shape[3]))
        tmp_full_imgs[0:full_imgs.shape[0],0:img_h,0:img_w,0:full_imgs.shape[3]] = full_imgs
        full_imgs = tmp_full_imgs
    if (leftover_w != 0):   #change dimension of img_w
        print ("the side W is not compatible with the selected stride of " +str(stride_w))
        print ("img_w " +str(img_w) + ", patch_w " +str(patch_w) + ", stride_w " +str(stride_w))
        print ("(img_w - patch_w) MOD stride_w: " +str(leftover_w))
        print ("So the W dim will be padded with additional " +str(stride_w - leftover_w) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_w+(stride_w - leftover_w),full_imgs.shape[3]))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:img_w,0:full_imgs.shape[3]] = full_imgs
        full_imgs = tmp_full_imgs
    print ("New full images shape: \n" +str(full_imgs.shape))
    return full_imgs


# Divide all the full_imgs in pacthes
def extract_ordered_overlap(full_imgs, patch_h, patch_w,stride_h,stride_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[3]==1 or full_imgs.shape[3]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[1]  #height of the full image
    img_w = full_imgs.shape[2] #width of the full image
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]
    print ("Number of patches on h : " +str(((img_h-patch_h)//stride_h+1)))
    print ("Number of patches on w : " +str(((img_w-patch_w)//stride_w+1)))
    print ("number of patches per image: " +str(N_patches_img) +", totally for this dataset: " +str(N_patches_tot))
    patches = np.empty((N_patches_tot,patch_h,patch_w,full_imgs.shape[3]))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                patch = full_imgs[i,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w,:]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches  #array with all the full_imgs divided in patches


def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape)==4)  #4D arrays
    assert (preds.shape[3]==1 or preds.shape[3]==3)  #check the channel is 1 or 3
    patch_h = preds.shape[1]
    patch_w = preds.shape[2]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    print ("N_patches_h: " +str(N_patches_h))
    print ("N_patches_w: " +str(N_patches_w))
    print ("N_patches_img: " +str(N_patches_img))
    assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    print ("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
    full_prob = np.zeros((N_full_imgs,img_h,img_w,preds.shape[3]))  #itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs,img_h,img_w,preds.shape[3]))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w,:]+=preds[k]
                full_sum[i,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w,:]+=1
                k+=1
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)  #at least one
    final_avg = full_prob/full_sum
    # assert(np.max(final_avg)<=1.0) #max value for a pixel is 1.0
    # assert(np.min(final_avg)>=0.0) #min value for a pixel is 0.0
    return final_avg


# Recompone the full images with the patches
def recompone(data,N_h,N_w):
    assert (data.shape[3]==1 or data.shape[3]==3)  #check the channel is 1 or 3
    assert(len(data.shape)==4)
    N_patch_per_img = N_w*N_h
    assert(data.shape[0]%N_patch_per_img == 0)
    N_full_imgs = data.shape[0]//N_patch_per_img
    patch_h = data.shape[1]
    patch_w = data.shape[2]
    N_patch_per_img = N_w*N_h
    #define and start full recompone
    full_recomp = np.empty((N_full_imgs,N_h*patch_h,N_w*patch_w,data.shape[3]))
    k = 0  #iter full img
    s = 0  #iter single patch
    while (s<data.shape[0]):
        #recompone one:
        single_recon = np.empty((N_h*patch_h,N_w*patch_w,data.shape[3]))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w,:]=data[s]
                s+=1
        full_recomp[k]=single_recon
        k+=1
    assert (k==N_full_imgs)
    return full_recomp


# Extend the full images because patch divison is not exact
def paint_border(data,patch_h,patch_w):
    assert (len(data.shape)==4)  #4D arrays
    assert (data.shape[1]==1 or data.shape[1]==3)  #check the channel is 1 or 3
    img_h=data.shape[2]
    img_w=data.shape[3]
    new_img_h = 0
    new_img_w = 0
    if (img_h%patch_h)==0:
        new_img_h = img_h
    else:
        new_img_h = ((int(img_h)/int(patch_h))+1)*patch_h
    if (img_w%patch_w)==0:
        new_img_w = img_w
    else:
        new_img_w = ((int(img_w)/int(patch_w))+1)*patch_w
    new_data = np.zeros((data.shape[0],data.shape[1],new_img_h,new_img_w))
    new_data[:,:,0:img_h,0:img_w] = data[:,:,:,:]
    
    return new_data


# Return only the pixels contained in the FOV, for both images and masks
def pred_only_FOV(data_imgs,data_masks,original_imgs_border_masks):
    assert (len(data_imgs.shape)==4 and len(data_masks.shape)==4)  #4D arrays
    assert (data_imgs.shape[0]==data_masks.shape[0])
    assert (data_imgs.shape[1]==data_masks.shape[1])
    assert (data_imgs.shape[2]==data_masks.shape[2])
    assert (data_imgs.shape[3]==1 and data_masks.shape[3]==1)  #check the channel is 1
    height = data_imgs.shape[1]
    width = data_imgs.shape[2]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE(i,y,x,original_imgs_border_masks)==True:
                    new_pred_imgs.append(data_imgs[i,y,x,:])
                    new_pred_masks.append(data_masks[i,y,x,:])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    
    return new_pred_imgs, new_pred_masks


# Function to set to black everything outside the FOV, in a full image
def kill_border(data, original_imgs_border_masks):
    assert (len(data.shape)==4)  #4D arrays
    assert (data.shape[3]==1 or data.shape[3]==3)  #check the channel is 1 or 3
    height = data.shape[1]
    width = data.shape[2]
    for i in range(data.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE(i,y,x,original_imgs_border_masks)==False:
                    data[i,y,x,:]=0.0


def inside_FOV_DRIVE(i,y,x,DRIVE_masks):
    assert (len(DRIVE_masks.shape)==4)  #4D arrays
    assert (DRIVE_masks.shape[3]==1)  #DRIVE masks is black and white
    # DRIVE_masks = DRIVE_masks/255.  #NOOO!! otherwise with float numbers takes forever!!

    if (x >= DRIVE_masks.shape[2] or y >= DRIVE_masks.shape[1]): #my image bigger than the original
        return False

    if (DRIVE_masks[i,y,x,0]>0):  #0==black pixels
        # print DRIVE_masks[i,y,x,0]  #verify it is working right
        return True
    else:
        return False
