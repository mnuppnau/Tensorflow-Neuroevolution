
import ast
from typing import Union, Any
from configparser import ConfigParser

import tensorflow as tf
from PyQt5 import QtWidgets

from tfne.visualizer import TFNEVWelcomeWindow

import os
import pandas as pd
import tensorflow as tf
import zipfile
import urllib.request as request
import cv2
import numpy as np

class ChexpertSmallTF:
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.url = 'http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip'  # Replace with the actual URL
        self.dir_name = 'CheXpert-v1.0-small'  # Replace with the actual directory name
        self.data = None
        
        self.attr_all_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                      'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                      'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                      'Fracture', 'Support Devices']
        
        self.attr_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        
        self._maybe_download_and_extract()
        self._maybe_process(None)  # No data filter for now

    def _maybe_download_and_extract(self):
        fpath = os.path.join(self.root, os.path.basename(self.url))
        data_path = os.path.join(self.root, self.dir_name)
        if not os.path.exists(data_path):
            if not os.path.exists(fpath):
                print('Downloading ' + self.url + ' to ' + fpath)
                request.urlretrieve(self.url, fpath)
            print('Extracting ' + fpath)
            with zipfile.ZipFile(fpath, 'r') as z:
                z.extractall(self.root)
            os.unlink(fpath)
            print('Dataset extracted.')
            
    def _maybe_process(self, data_filter):
        processed_train_file = os.path.join(self.root, self.dir_name, 'processed_train.csv')
        processed_valid_file = os.path.join(self.root, self.dir_name, 'processed_valid.csv')
        
        if not (os.path.exists(processed_train_file) and os.path.exists(processed_valid_file)):
            valid_df = pd.read_csv(os.path.join(self.root, self.dir_name, 'valid.csv'))
            train_df = pd.read_csv(os.path.join(self.root, self.dir_name, 'train.csv'))
            
            train_df[self.attr_names] = train_df[self.attr_names].fillna(0)
            valid_df[self.attr_names] = valid_df[self.attr_names].fillna(0)
            
            train_df.to_csv(processed_train_file, index=False)
            valid_df.to_csv(processed_valid_file, index=False)

def preprocess_image(image_path, resize=None, center_crop_size=320):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if resize:
        image = cv2.resize(image, (resize, resize))
        
    y, x = image.shape
    start_x = x // 2 - (center_crop_size // 2)
    start_y = y // 2 - (center_crop_size // 2)
    
    cropped_image = image[start_y:start_y+center_crop_size, start_x:start_x+center_crop_size]
    normalized_image = cropped_image.astype(np.float32) / 255.0
    mean = 0.5330
    std = 0.0349
    normalized_image = (normalized_image - mean) / std
    
    expanded_image = np.expand_dims(normalized_image, axis=-1)
    expanded_image = np.repeat(expanded_image, 3, axis=-1)
    
    return expanded_image

def _parse_function(image_path, label, root_dir, resize=None):
    image_path = os.path.join(root_dir, image_path.decode("utf-8"))
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    if resize:
        image = tf.image.resize(image, [resize, resize])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def debug_function(img, lbl):
    print("Debug - img: ", img)
    print("Debug - lbl: ", lbl)
    return img, lbl

def load_chexpert_data(root_dir='./data', batch_size=32, resize=None):
    print("root dir : ", root_dir)
    dataset = ChexpertSmallTF(root=root_dir)
    train_df = pd.read_csv(os.path.join(root_dir, dataset.dir_name, 'processed_train.csv'))
    valid_df = pd.read_csv(os.path.join(root_dir, dataset.dir_name, 'processed_valid.csv'))
    
    # Modify dataframes
    train_df.replace(-1, 1, inplace=True)
    valid_df.replace(-1, 1, inplace=True)
    
    # Extract paths and labels
    #train_image_paths = train_df['Path'].values
    train_image_paths = [os.path.join('./data', p) for p in train_df['Path'].values]
    
    train_labels = train_df[dataset.attr_names].values.astype('int')
    
    #test_image_paths = valid_df['Path'].values
    test_labels = valid_df[dataset.attr_names].values.astype('int')
    test_image_paths = [os.path.join('./data', p) for p in valid_df['Path'].values]


    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))

    # Define a function to load and preprocess images
    def load_and_preprocess(image_path, label):
        print("image path : ", image_path)
        image = tf.image.decode_jpeg(tf.io.read_file(image_path))
        image = tf.image.grayscale_to_rgb(image,name=None)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image,[224,224])
        #image = tf.image.per_image_standardization(image)
        print("tensor shape : ", tf.shape(image))
        return image, label

    # Apply the function to the dataset
    train_dataset = train_dataset.map(load_and_preprocess)
    train_dataset = train_dataset.shuffle(buffer_size=2048) 
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(load_and_preprocess)
    test_dataset = test_dataset.shuffle(buffer_size=2048)
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    # Pull a single batch from the training dataset
    #for image_batch, label_batch in train_dataset.take(1):
    #    print("Image batch shape: ", image_batch.shape)
    #    print("Label batch shape: ", label_batch.shape)
    
    #    # Print the first few values of the first image in the batch
    #    print("First few pixel values of the first image in batch: ", image_batch[0].numpy()[0][0])
    
    #    # Print the first label in the batch
    #    print("First label in batch:", label_batch[0].numpy())

    return train_dataset, test_dataset

def parse_configuration(config_path) -> object:
    """
    Takes a configuration file path, reads the configuration in ConfigParser and then returns it.
    @param config_path: string of configuration file path
    @return: ConfigParser instance
    """
    config = ConfigParser()
    with open(config_path) as config_file:
        config.read_file(config_file)

    return config


def read_option_from_config(config, section, option) -> Any:
    """
    @param config: ConfigParser instance
    @param section: string of the config section to read from
    @param option: string of the config section option to read
    @return: literal evaluated value of the config option
    """
    value = ast.literal_eval(config[section][option])
    print("Config value for '{}/{}': {}".format(section, option, value))
    return value

def select_random_value(min_val, max_val, step, stddev, threshold=4):
    """
    Selects a random value based on min, max, step, and stddev.
    
    :param min_val: Minimum value for the parameter
    :param max_val: Maximum value for the parameter
    :param step: Step size between possible values
    :param stddev: Standard deviation for the normal distribution
    :return: Selected random value
    """
    # Calculate the mean (you can adjust this if you want a different distribution)
    mean = (max_val + min_val) / 2

    # Generate a value from a normal distribution
    initial_value = np.random.normal(loc=mean, scale=stddev)

    # Use round_with_step function to get the final value
    final_value = round_with_step(initial_value, min_val, max_val, step)
    
    # Check if final_value is below the threshold
    if final_value < threshold:
        final_value = threshold

    return final_value

def round_with_step(value, minimum, maximum, step) -> Union[int, float]:
    """
    @param value: int or float value to round
    @param minimum: int or float specifying the minimum value the rounded result can take
    @param maximum: int or float specifying the maximum value the rounded result can take
    @param step: int or float step value of which the rounded result has to be a multiple of.
    @return: Rounded int or float (identical type to 'value') that is a multiple of the supplied step
    """
    lower_step = int(value / step) * step
    if value % step - (step / 2.0) < 0:
        if minimum <= lower_step <= maximum:
            return lower_step
        if lower_step < minimum:
            return minimum
        if lower_step > maximum:
            return maximum
    else:
        higher_step = lower_step + step
        if minimum <= higher_step <= maximum:
            return higher_step
        if higher_step < minimum:
            return minimum
        if higher_step > maximum:
            return maximum


def start_visualizer(tfne_state_backup_dir_path=None):
    """
    Starts TFNE visualizer and optionally takes the directory path of the TFNE state backup
    @param tfne_state_backup_dir_path: Optional string specifying the directory of the TFNE state backup
    """
    tfnev = QtWidgets.QApplication(sys.argv)
    tfnev_welcomewindow = TFNEVWelcomeWindow(tfne_state_backup_dir_path)
    tfnev.exec()


def set_tensorflow_memory_growth():
    """
    Set memory growth to true in the Tensorflow backend, fixing memory allocation problems that can occur on NVIDIA
    Turing GPUs
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
