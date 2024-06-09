import os
import numpy as np
import tensorflow as tf
from utils.helper_functions import prepareTrainDict, create_patches


class CustomDataGenerator(tf.keras.utils.Sequence):
    'Custom Data Generator based on TensorFlow'
    def __init__(self, 
                 img_dir, 
                 msk_dir, 
                 img_size=(224,224), 
                 batch_size=8, 
                 num_img_channel=3, 
                 num_msk_channel=1,
                 norm_factor_img=255, 
                 norm_factor_msk=255, 
                 num_class=1, 
                 is_train=True,
                 patchify=False,
                 patch_shape=(64,64),
                 overlap_ratio=0.5,
                 deep_supervision=False,
                 model_depth=5,
                 ds_type='UNet'):
        'Initialization'
        self.img_dir = img_dir
        self.img_list = os.listdir(self.img_dir)
        self.msk_dir = msk_dir
        self.msk_list = os.listdir(self.msk_dir)
        assert len(self.img_list) == len(self.msk_list), "Number of Images and Masks should be the same!"
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_img_channel = num_img_channel
        self.num_msk_channel = num_msk_channel
        self.norm_factor_img = norm_factor_img
        self.norm_factor_msk = norm_factor_msk
        self.num_class = num_class
        self.is_train = is_train
        self.patchify = patchify
        self.patch_shape = patch_shape
        self.overlap_ratio = overlap_ratio
        self.deep_supervision = deep_supervision
        self.model_depth = model_depth
        self.ds_type = ds_type
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return np.int_(np.math.ceil(len(self.img_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of names of Images
        list_IDs_temp = [self.img_list[k] for k in indexes]
        # Generate data
        x, y = self.__data_generation(list_IDs_temp)
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_list))
        if self.is_train == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, dim, n_channels)
        # Initialization
        if self.patchify == True:
            img_patch_counter = 0
            msk_patch_counter = 0
            x = np.empty((10000, self.patch_shape[0], self.patch_shape[1], self.num_img_channel), dtype=np.float32)
            y = np.empty((10000, self.patch_shape[0], self.patch_shape[1], self.num_msk_channel), dtype=np.int8)
        else:
            x = np.empty((self.batch_size, self.img_size[0], self.img_size[1], self.num_img_channel), dtype=np.float32)
            y = np.empty((self.batch_size, self.img_size[0], self.img_size[1], self.num_msk_channel), dtype=np.int8)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Load Image
            img = tf.keras.utils.load_img(path=f'{self.img_dir}/{ID}',
                                            color_mode='rgb',
                                            target_size=self.img_size,
                                            interpolation='lanczos',
                                            keep_aspect_ratio=False
                                            ) / self.norm_factor_img
            # img = img / self.norm_factor_img
            if self.patchify == True:
                patches, num_patches = create_patches(img, self.patch_shape, self.overlap_ratio)
                x[img_patch_counter:img_patch_counter + num_patches,:,:,:] = np.reshape(np.squeeze(patches), (num_patches, self.patch_shape[0], self.patch_shape[1], self.num_img_channel))
                img_patch_counter = img_patch_counter + num_patches
            else: 
                x[i,:,:,:] = img
            # Load Corresponding Mask
            msk = tf.keras.utils.load_img(path=f'{self.msk_dir}/{ID}',
                                            color_mode='grayscale',
                                            target_size=self.img_size,
                                            interpolation='nearest',
                                            keep_aspect_ratio=False
                                            )
            msk = msk / self.norm_factor_msk
            if self.patchify == True:
                patches, num_patches = create_patches(msk, self.patch_shape, self.overlap_ratio)
                y[msk_patch_counter:msk_patch_counter + num_patches,:,:,0] = np.reshape(np.squeeze(patches), (num_patches, self.patch_shape[0], self.patch_shape[1], self.num_msk_channel))
                msk_patch_counter = msk_patch_counter + num_patches
            else: 
                y[i,:,:,0] = msk
        if self.patchify == True:
            x = x[0:img_patch_counter,:,:,:]
            y = y[0:msk_patch_counter,:,:,:]
        if self.deep_supervision == True:
            y = prepareTrainDict(y, self.model_depth, self.ds_type)
        return x, y
    