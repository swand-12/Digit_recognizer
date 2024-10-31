# Digit Recognizer

This project uses the **MNIST dataset** to create a digit recognition model, which can recognize handwritten digits from 0 to 9. A **Convolutional Neural Network (CNN)** model was trained to classify these digits, and the model is deployed as a web application using **Streamlit**.

## Try the App

You can try out the digit recognizer here: [Digit Recognizer on Streamlit](http://digitrecognizerr.streamlit.app/)

## Project Overview

The goal of this project was to build a machine learning model capable of recognizing handwritten digits, which could be used for digit-based applications, like postal code recognition, check processing, or similar OCR (Optical Character Recognition) tasks.

### The Dataset: MNIST

The **MNIST dataset** is a widely-used benchmark dataset in the field of machine learning. It consists of:
- **60,000 training images** and **10,000 testing images** of grayscale handwritten digits.
- Each image is **28x28 pixels** and labeled from 0 to 9.

This dataset is considered a beginner-friendly dataset, often used for demonstrating image processing and classification models in deep learning.

### Model Architecture: Convolutional Neural Network (CNN)

The model architecture for this digit recognizer is a **Convolutional Neural Network (CNN)**. CNNs are particularly effective for image classification because they can capture spatial hierarchies and patterns in the data. Here’s a breakdown of the model architecture:

1. **Convolutional Layers**: 
   - Three convolutional layers with batch normalization and ReLU activations.
   - The first layer has **16 filters**, the second **32 filters**, and the third **64 filters**—all with a 3x3 kernel and padding to maintain spatial dimensions.
   
2. **Pooling Layers**: 
   - Max pooling layers follow each convolutional block to reduce spatial dimensions and retain important features.

3. **Dropout Layer**:
   - A dropout layer is included with a probability of 0.5 to prevent overfitting by randomly turning off some neurons during training.

4. **Fully Connected Layers**:
   - A dense layer with 256 neurons and a ReLU activation.
   - An output layer with 10 neurons (one for each digit class) for the final classification.
