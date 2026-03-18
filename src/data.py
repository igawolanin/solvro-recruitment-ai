import os
from PIL import Image
import numpy as np
from src.config import BaseConfig
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def load_images(path):
    """
    Load images from path directory to a NumPy array

    Parameters:
    - path (str): Path to the directory where the folders will images are.

    Raises:
    - ValueError: If error while loading the data.

    Returns:
    - data (np.array(uint8)): Array of 2D images.
    - labels (np.array(uint8)): Array of labels corresponding to images in data.
    - class_to_num (dict): Class name mapped to an integer.
    """
    i = 0
    data = []
    labels = []
    class_to_num = {}
    for directory in sorted(os.listdir(path)):
        path_to_files = os.path.join(path, directory)
        for file in sorted(os.listdir(path_to_files)):
            img_path = os.path.join(path_to_files, file)
            img = Image.open(img_path)
            img = np.asarray(img)
            data.append(img)
            if directory not in class_to_num:
                class_to_num[directory] = i
                i+=1
            labels.append(class_to_num[directory])

    if len(data) != len(labels):
        raise ValueError("Error with loading data")

    return np.array(data), np.array(labels), class_to_num

def split_data(data, labels):
    """
    Split dataset into train, validation and test sets.

    Parameters:
    - data (np.array): Array of images.
    - labels (np.array): Array of labels corresponding to images.

    Returns:
    - X_train (np.array): Training images.
    - X_val (np.array): Validation images.
    - X_test (np.array): Test images.
    - y_train (np.array): Training labels.
    - y_val (np.array): Validation labels.
    - y_test (np.array): Test labels.
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(data, labels, test_size=BaseConfig.TEST_SIZE, stratify=labels, random_state=BaseConfig.SEED)
    val_size = BaseConfig.VAL_SIZE/(BaseConfig.TRAIN_SIZE + BaseConfig.VAL_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=BaseConfig.SEED)

    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_mean_std(data):
    """
    Compute mean and standard deviation of dataset for normalization.

    Parameters:
    - data (np.array): Array of images.

    Returns:
    - mean (float): Mean pixel value (in range 0–1).
    - std (float): Standard deviation of pixel values (in range 0–1).
    """
    data = data.astype(np.float32) / 255.0
    mean = data.mean()
    std = data.std()
    return mean, std

def get_test_transform(mean, std):
    """
    Create transformation pipeline for validation/test data.

    Applies:
    - conversion to tensor (0–1 range)
    - channel expansion (1 -> 3 channels)
    - normalization using dataset statistics

    Parameters:
    - mean (float): Mean pixel value.
    - std (float): Standard deviation of pixel values.

    Returns:
    - transform (torchvision.transforms.Compose): Transformation pipeline.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=(mean,mean,mean), std=(std,std,std))

    ])
    return transform

def get_train_transform(mean, std, is_flippable):
    """
    Create transformation pipeline for training data.

    Includes data augmentation and normalization.

    If is_flippable is True, horizontal flip augmentation is applied.

    Parameters:
    - mean (float): Mean pixel value.
    - std (float): Standard deviation of pixel values.
    - is_flippable (bool): Whether horizontal flipping is allowed for given class.

    Returns:
    - transform (torchvision.transforms.Compose): Transformation pipeline.
    """
    if is_flippable:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=(mean,mean,mean), std=(std,std,std))

        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=(mean,mean,mean), std=(std,std,std))

        ])
    return transform

class ImgDataset(Dataset):
    """
    Custom PyTorch Dataset for image classification.

    Handles:
    - returning image-label pairs
    - applying appropriate transforms depending on training mode
    - optional class-specific augmentation (flipping)

    Parameters:
    - data (np.array): Array of images.
    - labels (np.array): Array of labels.
    - is_train (bool): Whether dataset is used for training.
    - transform_flip (callable, optional): Transform for flippable classes.
    - transform_noflip (callable, optional): Transform for non-flippable classes.
    - transform_eval (callable, optional): Transform for validation/test data.
    - flippable_classes (iterable, optional): Set of class indices that can be flipped.
    """
    def __init__(self, data, labels, is_train, transform_flip=None, transform_noflip=None, transform_eval=None, flippable_classes=None):
        if len(data) != len(labels):
            raise ValueError("Dataset must have the same length as labels")
        self.data = data
        self.labels  = labels
        self.is_train = is_train
        self.transform_flip = transform_flip
        self.transform_noflip = transform_noflip
        self.transform_eval = transform_eval
        self.flippable_classes = flippable_classes

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        """
        Get a single sample from dataset.

        Applies appropriate transform depending on:
        - training/evaluation mode
        - whether class allows flipping

        Parameters:
        - idx (int): Index of sample.

        Returns:
        - sample (Tensor): Transformed image.
        - label (int): Corresponding label.
        """
        sample = self.data[idx]
        label = self.labels[idx]
        if self.is_train:
            if int(label) in self.flippable_classes:
                sample = self.transform_flip(sample)
            else:
                sample = self.transform_noflip(sample)
        else:
            sample = self.transform_eval(sample)
        return sample, int(label)
