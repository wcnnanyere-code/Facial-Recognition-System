# Facial Recognition System

A convolutional neural network (CNN) project for image-based facial classification using TensorFlow and Keras.

The system performs image preprocessing, data augmentation, CNN-based training and model evaluation using classification metrics and a confusion matrix.

---

## Overview

This project explores the use of deep learning and computer vision techniques for facial image classification.

A Convolutional Neural Network (CNN) is trained on labelled facial images and evaluated on a separate validation dataset.

The project includes:

* Image preprocessing
* Data augmentation
* CNN model development
* Model training
* Early stopping
* Model persistence
* Classification evaluation
* Confusion matrix analysis
* Training and validation performance visualisation

---

## Objectives

The main objectives of the project are to:

* Develop a CNN for facial image classification
* Preprocess and normalise image data
* Improve model generalisation using data augmentation
* Monitor training and validation performance
* Evaluate classification performance using multiple metrics
* Analyse classification errors using a confusion matrix

---

## Technologies

The project was developed using:

* Python
* TensorFlow
* Keras
* NumPy
* Scikit-learn
* Matplotlib

---

## Model Architecture

The CNN consists of three convolutional blocks followed by fully connected layers.

The architecture includes:

```text
Input Image
    │
    ▼
Conv2D (32 filters)
    │
    ▼
MaxPooling
    │
    ▼
Conv2D (64 filters)
    │
    ▼
MaxPooling
    │
    ▼
Conv2D (128 filters)
    │
    ▼
MaxPooling
    │
    ▼
Flatten
    │
    ▼
Dense (128 neurons)
    │
    ▼
Dropout (0.5)
    │
    ▼
Softmax Output
```

The input images are resized to **128 × 128 pixels**.

---

## Data Preprocessing

Images are normalised by scaling pixel values to the range between 0 and 1.

Training data uses data augmentation techniques including:

* Rotation
* Shearing
* Zoom
* Horizontal flipping

This helps expose the model to variations in the training images and can improve generalisation.

The validation data is normalised but is not augmented.

---

## Training

The model uses:

* Adam optimiser
* Categorical cross-entropy loss
* Accuracy as a training metric
* Early stopping
* Maximum of 50 training epochs
* Batch size of 32

Early stopping monitors validation loss and restores the best-performing model weights.

---

## Evaluation

The trained model is evaluated using:

* Classification accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Training and validation accuracy and loss are also visualised to analyse model behaviour during training.

---

## Results

The evaluation produces:

* Training and validation accuracy curves
* Training and validation loss curves
* Classification report
* Confusion matrix
* Test accuracy

The exact results depend on the dataset and training configuration used.

---

## Project Structure

```text
facial-recognition-system/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── facial_recognition.py
│
├── data/
│   ├── train/
│   └── validation/
│
└── results/
```

The dataset is not included in the public repository.

---

## Dataset

The image dataset used for training and evaluation is not included in this repository.

This is intentional to avoid redistributing a dataset whose licensing and usage permissions may not allow public redistribution.

To run the project locally, place the authorised dataset in:

```text
data/
├── train/
└── validation/
```

The directory structure should contain one subdirectory per class.

---

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Place the authorised training and validation images inside the appropriate directories.

### 3. Run the training script

```bash
python src/facial_recognition.py
```

The script trains the CNN, saves the trained model and generates evaluation outputs.

---

## GPU Support

The project checks whether a compatible GPU is available through TensorFlow.

If a compatible GPU is detected, TensorFlow can use it to accelerate model training.

---

## Limitations

The system has several limitations:

* Performance depends strongly on the quality and diversity of the training dataset.
* CNN-based classification may not generalise well to significantly different environments.
* Changes in lighting, pose and image quality can affect predictions.
* The system is intended as an academic machine learning project rather than a production biometric identification system.
* The dataset used for the original project is not redistributed through this repository.

---

## Future Improvements

Potential improvements include:

* Transfer learning using pretrained CNN architectures
* Face detection before classification
* Improved handling of different poses and lighting conditions
* Larger and more diverse datasets
* Additional performance evaluation
* Real-time camera-based inference
* Deployment as a web or desktop application

---

## Academic Context

This project was developed as part of my Software Engineering studies at the University of Greenwich.

It demonstrates practical experience with:

* Computer vision
* Deep learning
* CNN architectures
* Image preprocessing
* Model training
* Model evaluation
* Python-based machine learning

---

## Author

**Wesley Nnanyere**

Software Engineer

Portfolio: https://wndevs.com/
