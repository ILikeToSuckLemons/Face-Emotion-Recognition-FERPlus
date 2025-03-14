import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image  # Add PIL import

import outlier as op
from Datapipeline import DataPipelineParams, Augmentation, Dataset

FER_CLASS_MAPPING = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'sadness',
    5: 'surprise',
    6: 'neutral'
}

FER_PLUS_CLASS_MAPPING = {
    0: 'neutral',
    1: 'happiness',
    2: 'surprise',
    3: 'sadness',
    4: 'anger',
    5: 'disgust',
    6: 'fear',
    7: 'contempt'
}

COLUMN_NAMES = ['dataset', 'image', 'fer_code', 'neutral', 'happiness', 
                'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 
                'no-face']

IMG_SHAPE = 48  # Assuming grayscale images (1 channel)

def get_augmentation_transforms(augmentation):
    """Returns torchvision transformations for augmentation."""
    base_transforms = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]

    if augmentation == Augmentation.HIGH:
        extra_transforms = [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(IMG_SHAPE, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    elif augmentation == Augmentation.MEDIUM:
        extra_transforms = [
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2),
        ]
    else:
        extra_transforms = []

    return transforms.Compose(extra_transforms + base_transforms)

def get_data_pipeline(dataset_df, params, shuffle=False):
    """Prepares the dataset for training in PyTorch."""
    images, labels = _get_images_labels(dataset_df, params.dataset, params.cross_entropy, params.original_preprocessing)
    
    transform = get_augmentation_transforms(params.augmentation)
    
    dataset = FacialExpressionDataset(images, labels, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffle)
    
    print("Number of elements:", len(dataset))
    return dataloader

class FacialExpressionDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for facial expression images."""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert numpy array to PIL image before applying transformations
        image = Image.fromarray(image)  # Convert numpy.ndarray to PIL Image

        if self.transform:
            image = self.transform(image)

        return image, label

def get_fer_class_mapping():
    return FER_CLASS_MAPPING

def get_fer_plus_class_mapping():
    return FER_PLUS_CLASS_MAPPING

def _get_images_labels(dataset_df, dataset, cross_entropy, original_preprocessing):
    """Extracts image and label data from DataFrame."""
    if dataset == Dataset.FERPLUS and original_preprocessing:
        dataset_df = op.get_dataset_without_original_outliers(dataset_df, cross_entropy, COLUMN_NAMES)
    elif dataset == Dataset.FERPLUS:
        dataset_df = op.get_dataset_without_custom_outliers(dataset_df, COLUMN_NAMES)

    # Convert images from strings to numpy arrays
    image_data = np.empty((len(dataset_df), IMG_SHAPE, IMG_SHAPE), dtype=np.uint8)
    for i, img in enumerate(dataset_df['image']):
        image_data[i] = _str_to_image_data(img)

    # Convert labels
    if dataset == Dataset.FER:
        int_labels = dataset_df.iloc[:, 2].values
        if cross_entropy:
            label_data = int_labels  # Return integer labels for cross-entropy
        else:
            label_data = _basis_vectors(int_labels, 7)  # One-hot encoding for FER
    else:  # FER-Plus
        label_data = dataset_df.iloc[:, 3:].values
        if cross_entropy:
            label_data = label_data.argmax(axis=1)  # Convert to integer labels for cross-entropy
        else:
            int_labels = label_data.argmax(axis=1)
            label_data = _basis_vectors(int_labels, 8)  # One-hot encoding for FER-Plus

    return image_data, label_data


def _str_to_image_data(image_blob):
    """Converts a string-encoded image into a numpy array."""
    image_string = image_blob.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(IMG_SHAPE, IMG_SHAPE)
    return image_data

def _p_distribution(x):
    """Normalizes probability distributions."""
    return (x / x.sum()).tolist() if isinstance(x, np.ndarray) else [float(xi) / sum(x) for xi in x]

def _basis_vectors(int_labels, n_classes):
    """Converts integer labels into one-hot encoding."""
    label_data = np.zeros((len(int_labels), n_classes))
    label_data[np.arange(len(int_labels)), int_labels] = 1
    return label_data
