import torch
import matplotlib.pyplot as plt# for visualising image and plots
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report#for evaluating the model
import numpy as np#for calculations
from torchvision import datasets, transforms#used to load and preprocess the MNIST dataset
from collections import Counter#to count the frequency
import torch.nn as nn#for building the neural network layers
import torch.nn.init as init#for initializing
from torch.utils.data import DataLoader#load data
import torch.optim as optim#optimizes

#model national institute of standard and technology database

#convert images to tensors and normalize using MNIST's mean and std
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])#converts PIL(python imaging library) to tensors

# Load MNIST training data with transformation applied
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Display 25 sample digit images from the dataset
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
axes = axes.ravel()
for i in range(15):
    img, lbl = mnist[i]
    axes[i].imshow(img.squeeze(), cmap='Blues')#removes the single dimension channels
    axes[i].set_title(lbl)
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# Count how many times each digit appears in the dataset
labels = mnist.targets
digit_counts = Counter(labels.numpy())
lbls = sorted(digit_counts.keys())
count = [digit_counts[i] for i in lbls]

#prints the count and %age of each digit class
sample = len(labels)
print("Digit Counts:", digit_counts)
print("\n% Distribution")
for i in sorted(digit_counts.keys()):
    percent = (digit_counts[i] / sample) * 100
    print(f"{i} : {percent:.2f}%")

# Visualize digit distribution using a bar plot
plt.figure(figsize=(8, 6))
plt.bar(lbls, count, color='skyblue')
plt.xlabel('Digits')
plt.ylabel('Count')
plt.title('Distribution of Digits')
plt.show()

# defining a CNN model for MNIST digit classification with 2 conv layers
#maxpooling to downdsample feature maps
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)#to prevent overfitting
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)#10 neurons for each digit
        self._initialize_weights()#uses Xavier initialization

    def forward(self, x):
        x=torch.relu(self.conv1(x))
        x=self.maxpool(x)
        x=self.dropout1(x)
        x=torch.relu(self.conv2(x))
        x=self.maxpool(x)
        x=self.dropout1(x)
        x=x.view(-1, 32 * 7 * 7)
        x=torch.relu(self.fc1(x))
        x=self.dropout2(x)
        x=self.fc2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.01)

# Load training and testing datasets with transformations
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders to efficiently load batches
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, define loss function and optimizer
model = MNIST_CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model for multiple epochs while recording performance
num_epochs = 15
train_losses, val_losses = [], []
train_accs, val_accs = [], []

#epochs refers to complete pass through the entire training dataset
for epoch in range(num_epochs):
    model.train()
    correct, total, running_loss = 0, 0, 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_losses.append(running_loss / len(train_loader))
    train_accs.append(100 * correct / total)

    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_losses.append(val_loss / len(test_loader))
    val_accs.append(100 * correct / total)

    print(f"Epoch {epoch+1}: Train Acc = {train_accs[-1]:.2f}%, Val Acc = {val_accs[-1]:.2f}%")

# Plot loss over epochs
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

# Plot accuracy over epochs
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()
plt.show()

# Evaluate model: get predictions on the test set
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

# Print precision, recall, F1-score for each digit class
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

# Display a confusion matrix to visualize misclassifications
#confusion matrix is used for classification problems and it helps to visualize how well a machine learning model is performing

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()