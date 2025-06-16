import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} - {'GPU' if device.type == 'cuda' else 'CPU'}")

# --- DATA PATH ---
data_dir = r"C:\Users\ROHIT\Downloads\Parkison MRI Datset\ntua-parkinson-dataset-master"

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --- LOAD DATA ---
dataset = datasets.ImageFolder(data_dir, transform=transform)
class_names = dataset.classes
print("Classes:", class_names)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

val_size = int(0.1 * len(train_data))
train_size = len(train_data) - val_size
train_data, val_data = random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# --- SOFT ATTENTION MODULE ---
class SoftAttention(nn.Module):
    def __init__(self, in_channels):
        super(SoftAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention

# --- DENSENET WITH ATTENTION ---
class DenseNetWithAttention(nn.Module):
    def __init__(self):
        super(DenseNetWithAttention, self).__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1
        base_model = densenet121(weights=weights)
        self.features = base_model.features
        self.attention = SoftAttention(1024)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- TRAINING ---
model = DenseNetWithAttention().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

num_epochs = 15
best_val_loss = float('inf')
patience = 3
patience_counter = 0
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    train_acc = 100 * correct / len(train_loader.dataset)
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    train_accuracies.append(train_acc)

    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()

    val_acc = 100 * val_correct / len(val_loader.dataset)
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}% | Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# --- TEST SET EVALUATION ---
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nClassification Report on Test Set:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# --- PLOTTING METRICS ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.tight_layout()
plt.show()

# --- DAT TO IMAGE CONVERSION ---
def dat_to_image(dat_path, shape=(224, 224)):
    data = np.fromfile(dat_path, dtype=np.uint8)
    image_array = data[:shape[0]*shape[1]].reshape(shape)
    if image_array.max() > 255:
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
        image_array = (image_array * 255).astype(np.uint8)
    img = Image.fromarray(image_array).convert("RGB")
    return img

# --- IMAGE PREDICTION FUNCTION ---
def predict_from_image_path(image_path):
    try:
        if not os.path.exists(image_path):
            return f"❌ File not found: {image_path}"

        img = Image.open(image_path).convert("RGB")
        image_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            return class_names[predicted.item()]
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# --- DAT FILE PREDICTION FUNCTION ---
def predict_from_dat_file(dat_path):
    try:
        img = dat_to_image(dat_path)
        image_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            return class_names[predicted.item()]
    except Exception as e:
        return f"❌ Error with DAT file: {str(e)}"

# --- SAMPLE USAGE ---

# Predict from PNG image
png_image = r"C:\Users\ROHIT\Downloads\Parkison MRI Datset\ntua-parkinson-dataset-master\PD Patients\Subject75\0.DAT\s1\001.png"
print("PNG Prediction:", predict_from_image_path(png_image))

