# MNIST-image-classifcation
This project focuses on building a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The goal is to implement an effective deep learning model using PyTorch, analyze its performance, and suggest improvements.

Key steps include:

Data Loading & Exploration
The MNIST dataset is loaded using torchvision.datasets.MNIST. Sample images are visualized, and the class distribution of digits (0–9) is analyzed using plots.

CNN Implementation
A custom CNN is designed with:

At least 2 convolutional layers

Activation functions (ReLU)

Pooling layers

A fully connected layer before the final output layer

An output layer with 10 neurons for digit classification

Model Training
The model is trained using an appropriate loss function and optimizer over at least 15 epochs. Training and validation accuracy/loss are tracked and visualized using line plots.

Evaluation & Analysis
After training, the model is evaluated on the test set using metrics like accuracy, precision, recall, and F1-score. Misclassified digits are analyzed using a confusion matrix, and potential improvements are discussed.

This project demonstrates how deep learning models can accurately classify image data and how performance can be systematically analyzed and improved.