import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.ndimage
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F

class Visualizer:
    """
    Utility class for visualizing dataset characteristics and training results.

    Provides methods for:
    - displaying sample images
    - analyzing class distribution
    - inspecting image properties (size, brightness, sharpness)
    - visualizing training progress

    Parameters:
    - data (np.array): Array of images.
    - labels (np.array): Array of labels corresponding to images.
    - num_to_class (dict): Mapping from numeric labels to class names.
    """
    def __init__(self, data, labels, num_to_class):
        self.data = data
        self.labels = labels
        self.num_to_class = num_to_class

    # Showing images and labels from a dataset in one row, choosing images from dataset randomly or sequentially from index 0
    def show_images(self, is_random=False, n=5):
        """
        Display sample images from dataset.

        Images can be selected randomly or sequentially from the beginning.

        Parameters:
        - is_random (bool, optional): Whether to select images randomly.
        - n (int, optional): Number of images to display.
        """
        n = min(n, len(self.data))
        plt.figure(figsize=(2 * n, 2))
        if is_random:
            images = random.sample(range(len(self.data)), n)
        else:
            images = list(range(n))
        for idx, val in enumerate(images):
            plt.subplot(1, n, idx + 1)
            plt.imshow(self.data[val], cmap='gray')
            plt.axis('off')
            plt.title(self.num_to_class[self.labels[val]])
        plt.suptitle('Example of:')
        plt.tight_layout()
        plt.show()

    def show_class_distribution(self):
        """
        Visualize class distribution as a bar chart.

        Shows percentage of samples for each class.
        """
        classes = pd.Series(self.labels).value_counts().sort_index()
        classes = classes / classes.sum() * 100
        class_names = [self.num_to_class[i] for i in classes.index]
        plt.figure(figsize=(4, 2))

        plt.bar(class_names, classes, color='pink')
        plt.xlabel("Class")
        plt.ylabel("Percentage")
        plt.title("Class distribution")
        plt.xticks(rotation=90)
        plt.show()

    def show_unique_images(self):
        """
        Display one representative image per class.

        Selects the first occurrence of each class in the dataset.
        """
        unique_vals, start_index = np.unique(self.labels, return_index=True)
        n = len(unique_vals)
        plt.figure(figsize=(2*n, 2))
        for i, idx in enumerate(start_index):
            plt.subplot(1,n, i+1)
            plt.imshow(self.data[idx], cmap='gray')
            plt.axis('off')
            plt.title(self.num_to_class[unique_vals[i]])
        plt.suptitle('Unique class images')
        plt.tight_layout()
        plt.show()

    def show_image_size_info(self):
        """
        Print information about image sizes in the dataset.

        If all images have the same size, prints that size.
        Otherwise, prints all unique sizes found.
        """
        img_sizes = [x.shape for x in self.data]
        unique = set(img_sizes)
        if len(unique) == 1:
            size = next(iter(unique))
            print(f"All images have the same size (H, W): {size}")
        else:
            print(f"Number of unique image sizes: {len(unique)}")
            print("format: H, W")
            for x in sorted(unique):

                print(x, end=", ")

    def show_sharpness_distribution(self):
        """
        Visualize distribution of image sharpness using Laplace variance.

        Higher variance indicates sharper images, lower variance indicates blurrier images.
        """
        sharpness_scores = []
        for img in self.data:
            img_float = img.astype(np.float32)
            laplacian_image = scipy.ndimage.laplace(img_float, mode='reflect')
            score = laplacian_image.var()
            sharpness_scores.append(score)
        sharpness_scores = np.array(sharpness_scores)

        plt.figure(figsize=(4,4))
        plt.hist(sharpness_scores, bins=30, color='pink')
        plt.xlabel("Laplace variance")
        plt.ylabel("Number of images")
        plt.title("Sharpness distribution using Laplace variance")
        plt.show()

    def show_blurriest_images(self, n=3):
        """
        Display images with the lowest sharpness scores.

        Parameters:
        - n (int, optional): Number of blurriest images to display.
        """
        sharpness_idx_scores = []
        for idx, img in enumerate(self.data):
            img_float = img.astype(np.float32)
            laplacian_image = scipy.ndimage.laplace(img_float, mode='reflect')
            score = laplacian_image.var()
            sharpness_idx_scores.append((idx, score))
        sharpness_idx_scores.sort(key=lambda x: x[1])

        for i in range(n):
            idx = sharpness_idx_scores[i][0]
            plt.subplot(1, n, i + 1)
            plt.imshow(self.data[idx], cmap='gray')
            plt.title(f"var={round(sharpness_idx_scores[i][1])}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()


    def show_sharpest_images(self, n=3):
        """
        Display images with the highest sharpness scores.

        Parameters:
        - n (int, optional): Number of sharpest images to display.
        """
        sharpness_idx_scores = []
        for idx, img in enumerate(self.data):
            img_float = img.astype(np.float32)
            laplacian_image = scipy.ndimage.laplace(img_float, mode='reflect')
            score = laplacian_image.var()
            sharpness_idx_scores.append((idx, score))
        sharpness_idx_scores.sort(key=lambda x: x[1], reverse=True)

        for i in range(n):
            idx = sharpness_idx_scores[i][0]
            plt.subplot(1, n, i + 1)
            plt.imshow(self.data[idx], cmap='gray')
            plt.title(f"var={round(sharpness_idx_scores[i][1])}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def show_images_near_threshold(self, threshold, n=3):
        """
        Display images with sharpness values closest to a given threshold.

        Useful for analyzing borderline cases between blurry and sharp images.

        Parameters:
        - threshold (float): Sharpness threshold value.
        - n (int, optional): Number of images to display.
        """
        results = []
        for idx, img in enumerate(self.data):
            img_float = img.astype(np.float32)
            laplacian_image = scipy.ndimage.laplace(img_float, mode='reflect')
            score = laplacian_image.var()
            distance = abs(threshold-score)
            results.append((idx,score,distance))
        results.sort(key=lambda x:x[2])

        for i in range(n):
            plt.subplot(1,n,i+1)
            plt.imshow(self.data[results[i][0]], cmap='gray')
            plt.title(f"var={round(results[i][1])}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()


    def show_brightness_distribution(self):
        """
        Visualize distribution of image brightness.

        Brightness is computed as the mean pixel value of each image.
        """
        px_means = []
        for img in self.data:
            px_means.append(img.mean())
        plt.figure(figsize=(4,4))
        plt.hist(px_means, bins=30, color='pink')
        plt.title('Brightness distribution measured by pixel value: 0 - black, 255 - white')
        plt.xlabel('Pixel mean')
        plt.ylabel('Number of images')
        plt.tight_layout()
        plt.show()

    def show_training_process(self, history):
        """
        Visualize training and validation metrics over epochs.

        Plots loss and accuracy for both training and validation sets.

        Parameters:
        - history (dict): Dictionary containing:
            - train_loss (list of float): Training loss per epoch.
            - val_loss (list of float): Validation loss per epoch.
            - train_acc (list of float): Training accuracy per epoch.
            - val_acc (list of float): Validation accuracy per epoch.
        """
        x = np.arange(1, len(history['train_loss'] ) + 1)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(x, history['train_loss'], label='Train loss', color='pink')
        plt.plot(x, history['val_loss'], label='Validation loss', color='orchid')
        plt.title("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x, history['train_acc'], label='Train accuracy', color='pink')
        plt.plot(x, history['val_acc'], label='Validation accuracy', color='orchid')
        plt.title("Accuracy")
        plt.legend()
        plt.show()

    def show_confusion_matrix(self, predictions, targets):
        """
            Display confusion matrix and compute accuracy.

            Parameters:
            - predictions (np.ndarray): Predicted class labels.
            - targets (np.ndarray): True class labels.
            """
        cm = confusion_matrix(targets, predictions)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
        cm_display.plot(cmap='RdPu')
        plt.title("Confusion Matrix")
        plt.gca().grid(False)
        plt.show()

        acc = (predictions == targets).mean()
        print(f"Test accuracy: {np.round(acc, 4)}")

    def show_misclassified_images(self, test_data, preds, labels, idxs, n):
        """
        Display misclassified images.

        Parameters:
        - test_data (np.ndarray): Array of test images.
        - preds (np.ndarray): Predicted labels.
        - labels (np.ndarray): True labels.
        - idxs (np.ndarray): Indices of misclassified samples.
        - n (int): Number of images to display.
        """
        n = min(n, len(idxs))

        plt.figure(figsize=(2*n, 2))
        for i, idx in enumerate(idxs[:n]):
            plt.subplot(1, n, i + 1)
            plt.imshow(test_data[idx], cmap='gray')
            plt.title(f"p:{self.num_to_class[preds[idx]]}\nt:{self.num_to_class[labels[idx]]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def preprocess_cam(self, cam):
        """
        Preprocess CAM for visualization.

        Applies ReLU to remove negative values, resizes the CAM to match
        the original image size, and normalizes values to the range [0, 1].

        Parameters:
        - cam (torch.Tensor): Raw CAM tensor of shape (H, W).

        Returns:
        - cam (torch.Tensor): Processed CAM tensor resized and normalized
          for visualization.
        """
        target_size = self.data.shape[-2:]
        cam = torch.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
        cam = cam.squeeze()
        cam = cam - cam.min()
        cam = cam/(cam.max() + 1e-8)

        return cam

    def show_cam(self, cam, img):
        """
        Visualize CAM on top of the original image.

        Displays the grayscale image with the CAM heatmap on top,
        highlighting regions important for model prediction.

        Parameters:
        - cam (torch.Tensor or np.ndarray): Processed CAM.
        - img (torch.Tensor or np.ndarray): Input image corresponding to CAM.

        Notes:
        - If tensors are provided, they are automatically converted to NumPy.
        - For multi-channel input, only the first channel is displayed.
        """
        if torch.is_tensor(cam):
            cam = cam.cpu().detach().numpy()
        if torch.is_tensor(img):
            img = img.cpu().detach().numpy()
        if len(img.shape) == 3:
            img = img[0]
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.imshow(cam, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.title("CAM visualization")
        plt.colorbar()
        plt.show()
