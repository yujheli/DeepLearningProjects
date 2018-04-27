import csv
import random
import pickle
import scipy
import scipy.misc
import skimage
import skimage.io
import skimage.transform
import numpy as np
from parameter import *
from os import mkdir
from os.path import join, basename, exists

from PIL import Image


def convert_tags_for_hw4(tags):
    hair = np.zeros(len(HAIR_TAGS))
    for idx, tag in enumerate(HAIR_TAGS):
        if tag in tags:
            hair[idx] = 1
    eyes = np.zeros(len(EYES_TAGS))
    for idx, tag in enumerate(EYES_TAGS):
        if tag in tags:
            eyes[idx] = 1
    return np.concatenate([hair, eyes])

def show_and_save(X, rows=1, name='default.png', save=True):
    assert X.shape[0]%rows == 0
    int_X = ((X+1)/2*255).clip(0,255).astype('uint8')
    int_X = int_X.reshape(rows, -1, image_size, image_size,3).swapaxes(1,2).reshape(rows*image_size,-1, 3)
    image = Image.fromarray(int_X)
    # display(image)
    if save: image.save(join(OUTPUT_PATH, name))

def load_images_from_float32():
    images = []
    images_path = sorted(glob(join(FACE_PATH, '*')), key=lambda x: int(basename(x).split('.')[0]))
    for image_path in images_path:
        images.append(np.load(image_path))

    return np.stack(images) * 2 - 1

def load_filtered_images_from_float32(labels):
    filtered_idx = []
    for i in range(labels.shape[0]):
        if np.max(labels[i]) == 1.0:
            filtered_idx.append(i)
    images = []
    for idx in filtered_idx:
        image_path = join(FACE_PATH, '{0}.npy'.format(idx))
        images.append(np.load(image_path))
    return np.stack(images) * 2 - 1, labels[filtered_idx]

def load_filtered_images_augmentation(labels):
    filtered_idx = []
    for i in range(labels.shape[0]):
        if np.max(labels[i]) == 1.0:
            filtered_idx.append(i)
    images = []
    for idx in filtered_idx:
        image_path = join(FACE_PATH, '{0}.npy'.format(idx))
        image_array = np.load(image_path)
        if random.random() > 0.5:
            image_array = np.fliplr(image_array)
        images.append(image_array)
    return np.stack(images) * 2 - 1, labels[filtered_idx]


def load_labels():
    return np.load('onehot_labels.npy')

def readSampleInfo(file_path):
    with open(file_path, "rb") as file:
        list_ = pickle.load(file)
        return list_

def getImageArray(image_file_path):
    image_array = skimage.io.imread(image_file_path)
    resized_image_array = skimage.transform.resize(image_array, (image_size, image_size))
    if random.random() > 0.5:
        resized_image_array = np.fliplr(resized_image_array)
    return resized_image_array.astype(np.float32)

def getTrainData(train_data):
    real_image = np.zeros((batch_size, image_size, image_size, 3), dtype=np.float32)
    caption = np.zeros((batch_size, caption_size), dtype=np.float32)
    image_file = []
    for i, sample_data in enumerate(train_data):
        real_image[i,:,:,:] = getImageArray(sample_data[1])
        caption[i,:] = sample_data[3].flatten()
        image_file.append(sample_data[1])
    wrong_image = np.roll(real_image, 1, axis=0)
    noise = np.asarray(np.random.uniform(-1, 1, [batch_size, noise_size]), dtype=np.float32)
    return real_image, wrong_image, caption, noise, image_file

def getTestData(test_data):
    caption = np.zeros((1, caption_size), dtype=np.float32)
    image_file = []
    for i, sample_data in enumerate(test_data):
        caption[i,:] = sample_data[3].flatten()
        image_file.append(sample_data[1])
    noise = np.asarray(np.random.uniform(-1, 1, [1, noise_size]), dtype=np.float32)
    return caption, noise, image_file
