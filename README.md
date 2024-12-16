
# Convolutional Neural Network for Image Classification

## Project Overview
This project demonstrates the use of a Convolutional Neural Network (CNN) for binary image classification (e.g., cats vs. dogs). It covers the complete workflow from data preprocessing to model training and making predictions on unseen data. The CNN is implemented using TensorFlow and Keras.

## Features
- Preprocessing of training and testing datasets using image augmentation and normalization.
- Building a CNN architecture with multiple layers: convolution, pooling, flattening, and dense layers.
- Training the model using a binary cross-entropy loss function and Adam optimizer.
- Making predictions on single images with a saved model.

## Requirements
To run this project, you need the following:
- Python 3.7 or higher
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- PIL

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/cnn-image-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd cnn-image-classification
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
- Place your training and testing datasets in the following structure:
  ```
  dataset/
  |-- training_set/
  |   |-- cats/
  |   |-- dogs/
  |-- test_set/
      |-- cats/
      |-- dogs/
  ```

## Usage
### Training the Model
1. Run the Jupyter Notebook or Python script to train the CNN:
   ```bash
   python train_model.py
   ```
2. The trained model will be saved as `cat_dog_classifier.keras`.

### Making Predictions
1. Use the saved model to make predictions on single images:
   ```bash
   python predict.py --image_path path_to_image.jpg
   ```
2. The script will output whether the image is a cat or a dog.

## Model Architecture
1. **Convolutional Layers**: Extract features using 32 and 64 filters.
2. **Pooling Layers**: Reduce spatial dimensions while retaining essential features.
3. **Flatten Layer**: Convert 2D features to a 1D vector.
4. **Fully Connected Layers**: Dense layers for final classification.
5. **Output Layer**: Sigmoid activation for binary output.

## Results
- Achieved an accuracy of XX% on the test dataset.
- Model can reliably classify cat and dog images with minimal error.

## https://medium.com/@p.kunal7997/from-pixels-to-predictions-building-a-cnn-for-image-classification-f4c190ce061f
