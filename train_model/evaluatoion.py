import json
import matplotlib.pyplot as plt
import seaborn as sns


with open("training_log.json", "r") as f:
    data = json.load(f)
    
train_losses = data["loss"]
train_accuracies = data["accuracy"]

# Vẽ loss và accuracy
plt.figure(figsize=(12, 5))

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

# Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Training Accuracy", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curve")
plt.legend()

plt.show()


