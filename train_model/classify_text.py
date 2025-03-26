import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
# from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchvision import transforms, datasets
from tqdm import tqdm
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inception import INCEPTION
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import torch

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = INCEPTION(num_classes=30).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Dataset
train_dir = "dataset/train1"
val_dir = "dataset/val1"

# Load Dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Training Loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, save_path="training_log.json"):
    model.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    
    # Save to JSON file
    with open(save_path, "w") as f:
        json.dump({"loss": train_losses, "accuracy": train_accuracies}, f)

    return train_losses, train_accuracies

# Evaluation Function
def evaluate_model(model, val_loader):
    model.eval()
    acc = Accuracy().to(device)
    precision = Precision(average='macro', num_classes=30).to(device)
    recall = Recall(average='macro', num_classes=30).to(device)
    f1 = F1Score(average='macro', num_classes=30).to(device)
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            acc.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            f1.update(preds, labels)
    
    print(f"Validation Accuracy: {acc.compute():.4f}")
    print(f"Validation Precision: {precision.compute():.4f}")
    print(f"Validation Recall: {recall.compute():.4f}")
    print(f"Validation F1 Score: {f1.compute():.4f}")


def plot_confusion_matrix(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(30), yticklabels=range(30))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()



# Get a batch of images from train_loader
images, labels = next(iter(train_loader))

# Show first 4 images
fig, axes = plt.subplots(1, 4, figsize=(10, 5))
for i in range(4):
    img = images[i].permute(1, 2, 0).numpy() 
    img = (img * 0.5) + 0.5  # Undo normalization
    axes[i].imshow(img)
    axes[i].axis("off")
plt.show()

# # Run Training and Evaluation
# train_model(model, train_loader, criterion, optimizer, num_epochs=10)
# evaluate_model(model, val_loader)
# plot_confusion_matrix(model, val_loader)
from tabulate import tabulate

params = []
for name, param in model.named_parameters():
    params.append([name, str(param.size()), param.requires_grad])

print(tabulate(params, headers=["Layer Name", "Size", "Requires Grad"], tablefmt="grid"))

