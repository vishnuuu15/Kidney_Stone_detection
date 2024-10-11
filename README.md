# Kidney Stone Detection Model using KidneyNet-V

This project aims to develop an efficient and lightweight kidney stone detection system using the KidneyNet-V model, which is based on the MobileNetV2 architecture. The system is capable of classifying CT images of kidneys into two categories: those with kidney stones and those without.

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

Kidney_Stone_Detection/  
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
│       └── Mark_VI_KidneyNetV.pth  
│  
├── Kidney_Stone_Detection_KidneyNetV_Model.ipynb  
├── Evaluating_KidneyNetV_Model.ipynb
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
- Streamlit (for the web interface)


## Dataset
The dataset is divided into training and testing sets, each containing two categories:

- Kidney_stone: CT images of kidneys with stones.
- Normal: CT images of kidneys without stones.


## Training the Model
The KidneyNet-V model is based on the MobileNetV2 architecture, optimized for lightweight and efficient performance. The training process involves data augmentation (rotation, flipping, zooming) to increase the model's robustness. The model was trained to maximize accuracy and its performance was evaluated using various metrics.  The trained model is saved in the `Saved_Models/models` directory.


## Using the Model
You can use the trained KidneyNet-V model to predict kidney stones in new CT images. The model takes an image file path as input, applies the necessary data preprocessing, and provides the prediction along with a visual output.

To make predictions, run the model using the provided Streamlit-based web interface. The user can upload an image to receive a real-time prediction on whether kidney stones are present.

## Evaluation Metrics
The model is evaluated using precision, recall, F1 score, and a confusion matrix. These metrics provide a comprehensive assessment of the model's performance on the test set.

## Results
After evaluating the model on the test set, the following metrics were obtained:

- Accuracy: 99.71%
- Precision: 0.9971
- Recall: 0.9971
- F1 Score: 0.9971

## Acknowledgments
This project utilizes the PyTorch framework and MobileNetV2 architecture, which were crucial for building and training the KidneyNet-V model. Special thanks to the creators and maintainers of these tools and libraries.