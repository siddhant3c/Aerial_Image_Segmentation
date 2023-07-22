# Aerial Image Segmentation Project

*A project completed under the **Coursera Project Network** by*


**Siddhant Chourasia**

B.Tech | Electrical Engineering | IIT Bombay

Certificate of Completion: [View Certificate](Coursera%20AT4Z2GMW5FEM.pdf)

---

## Overview

This project is a part of the **Coursera Project Network** and focuses on aerial image segmentation using the Massachusetts Roads Dataset. The objective is to develop a segmentation model capable of accurately identifying roads in aerial images. This can be useful for various applications, such as road network analysis and urban planning.



## Dataset

The dataset used in this guided project is a subset of the original Massachusetts Roads Dataset. It consists of 200 aerial images and their corresponding masks. Each image is of size 1500x1500 pixels, covering an area of 2.25 square kilometers.

The full dataset, which contains 1171 aerial images, can be found in the following link: [Full Dataset](https://www.cs.toronto.edu/~vmnih/data/).

## Environment Setup

The project is implemented in a Colab GPU runtime environment. The necessary packages are installed using the following commands:

```python
!pip install segmentation-models-pytorch
!pip install -U git+https://github.com/albumentations-team/albumentations
!pip install --upgrade opencv-contrib-python
```

## Configurations

Several configurations are set for the project:

- **CSV_FILE**: Path to the CSV file containing image and mask file paths.
- **DATA_DIR**: Directory containing the dataset.
- **DEVICE**: Device to run the model (e.g., 'cuda' for GPU).
- **EPOCHS**: Number of training epochs.
- **LR**: Learning rate for the optimizer.
- **BATCH_SIZE**: Batch size for data loading.
- **IMG_SIZE**: Size of the input images.
- **ENCODER**: Encoder architecture for the segmentation model.
- **WEIGHTS**: Pretrained weights for the encoder.

## Augmentation

Data augmentation is performed to increase the diversity of the training dataset. Albumentations library is used for image transformations. The augmentation functions include resizing and horizontal/vertical flipping.

## Custom Dataset

A custom dataset class, `SegmentationDataset`, is created to load and preprocess the images and masks. The dataset class returns preprocessed image and mask tensors.

## DataLoader

PyTorch DataLoader is used to load the custom datasets into batches. Separate loaders are created for training and validation datasets.

## Segmentation Model

The segmentation model is implemented using the Unet architecture from the segmentation-models-pytorch library. The model consists of an encoder network, which uses a pre-trained EfficientNet-B0 as the backbone, and a decoder network.

## Train and Validation Functions

The training and validation functions are defined to train and evaluate the model. The training function includes the forward pass, loss calculation, and backpropagation. The validation function evaluates the model's performance on the validation dataset.

## Training the Model

The model is trained using the Adam optimizer. The training process involves iterating through the training data for a specified number of epochs. The best model is saved based on the validation loss, which ensures that the model with the lowest validation loss is retained.

## Inference

To perform inference, the trained model is loaded, and sample images from the validation set are processed. The model predicts segmentation masks for the input images, which are then compared with the ground truth masks to visualize the model's performance.

*Note: The complete code is available in the actual implementation, but for brevity, only the main steps and explanations are provided in this report.*
