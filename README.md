# Kidney Stone Detection Model

This project aims to develop an automated kidney stone detection system using deep learning and visual explanation techniques. The system is built using the Fastai library with a ResNet34 architecture. It can classify CT images of kidneys into two categories: those with kidney stones and those without.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Training the Model](#training-the-model)
5. [Using the Model](#using-the-model)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Results](#results)
8. [Acknowledgments](#acknowledgments)

## Project Structure

The project directory is organized as follows:

Kidney_Stone_Detection_Model/
│
├── Dataset/  
│   ├── Train/  
│   │   ├── Kidney_stone/  
│   │   └── Normal/  
│   ├── Test/  
│   │   ├── Kidney_stone/  
│   │   └── Normal/  
│  
├── Saved_Models/  
│   └── models/  
│       └── Mark_IV_epoch_10.pth  
│  
├── Kidney_stone_detection_model.ipynb  
└── README.md  


- **Dataset:** Contains the training and testing images, organized into `Kidney_stone` and `Normal` categories.
- **Saved_Models:** Contains the trained models.
- **Kidney_stone_detection_model.ipynb:** Jupyter notebook for training, testing, and evaluating the model.
- **README.md:** This detailed explanation of the project.

## Requirements

To run this project, you need to have the following installed:

- Python 3.x
- Fastai
- PyTorch
- scikit-learn
- Matplotlib


## Dataset
The dataset is divided into training and testing sets, each containing two categories:

- Kidney_stone: CT images of kidneys with stones.
- Normal: CT images of kidneys without stones.


## Training the Model
The model is trained using the Fastai library with a ResNet34 architecture. The training process includes data augmentation and uses accuracy as the metric. The trained model is saved in the `Saved_Models/models` directory.


## Using the Model
You can use the trained model to predict kidney stones in new images. The model takes an image file path as input, performs necessary data augmentation, and displays the prediction along with the input image.

## Evaluation Metrics
The model is evaluated using precision, recall, F1 score, and a confusion matrix. These metrics provide a comprehensive assessment of the model's performance on the test set.

## Results
After evaluating the model on the test set, the following metrics were obtained:

- Precision: 0.9383
- Recall: 0.9364
- F1 Score: 0.9362

## Acknowledgments
This project utilizes the Fastai library and ResNet34 architecture, both of which are instrumental in building and training the deep learning model for kidney stone detection. Special thanks to the creators and maintainers of these tools and libraries.