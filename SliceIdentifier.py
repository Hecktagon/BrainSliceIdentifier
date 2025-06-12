#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:39:25 2025

@author: jasonwells
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
from torchvision import models
from sys import argv

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cudnn backend deterministic (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# === Configure 'device' ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === HELPER: Remove hidden/system files like .DS_Store, ._ etc. ===
def clean_file_list(file_list):
    return [f for f in file_list if not (f.startswith('.') or f.startswith('._'))]

# === Image transform ===
transform = transforms.Compose([
    transforms.Grayscale(),                # Ensure image is 1 channel
    transforms.Resize((224, 224)),         # Resize to match model input
    transforms.ToTensor(),                 # Convert to tensor
    transforms.Normalize([0.5], [0.5])     # Normalize to [-1, 1] range
])

# === DATASET for Triplets ===
class TripletDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.transform = transform
        self.folder = folder
        self.image_files = clean_file_list(sorted(os.listdir(folder)))
        self.images = [os.path.join(folder, f) for f in self.image_files]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        anchor_path = self.images[idx]
        anchor_img = Image.open(anchor_path).convert("L")
        if self.transform:
            anchor_img = self.transform(anchor_img)

        pos_idx = idx + 1 if idx + 1 < len(self.images) else idx
        pos_path = self.images[pos_idx]
        pos_img = Image.open(pos_path).convert("L")
        if self.transform:
            pos_img = self.transform(pos_img)

        neg_idx = (idx + 10) % len(self.images)
        neg_path = self.images[neg_idx]
        neg_img = Image.open(neg_path).convert("L")
        if self.transform:
            neg_img = self.transform(neg_img)

        return (anchor_img, pos_img, neg_img)

# === MODEL with dynamic linear size ===
class ResNetEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embedding = nn.Linear(resnet.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x


# === Triplet Loss ===
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)


# === Training function ===
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    count = 0

    for anchor, positive, negative in dataloader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        loss = criterion(anchor_out, positive_out, negative_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    if count == 0:
        return 0
    return total_loss / count


# === Prepare reference embeddings ===
def load_and_preprocess(path, transform):
    img = Image.open(path).convert("L")
    return transform(img).unsqueeze(0).to(device)


# === Calculate distances and get top 3 matches ===
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


# === Make a uniform scale vector out of measurements ===
def normalize_shape(width, height, whole):
    vec = np.array([width, height, whole], dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.zeros_like(vec)
    return vec / norm


# === calculate a 'distance' between query and reference measurements ===
def shape_difference(a_dims, b_dims):
    a_norm = normalize_shape(*a_dims)
    b_norm = normalize_shape(*b_dims)
    similarity = np.dot(a_norm, b_norm)  # Cosine similarity
    return 1 - similarity  # 0 means identical shape, 1 means totally different


# === uses filepath of format '<filename> <dataset>_<slice>.jpg' to determine dataset and slice ===
def get_dataset_and_slice(path):
    # grabbing just the image filename
    image = path.split("/")[-1]
    # getting the '<dataset>_<slice>' part of the filename, ex: 'PA2_002'
    data_and_slice = image.split(".")[0].split(" ")[-1]
    dataset = int(data_and_slice[2])
    slice_num = int(data_and_slice[-3:])

    return dataset, slice_num


def main():
    # === CONFIGURATION ===
    reference_folder = 'Master_Reference_Folder'
    measurement_folder = 'Hippocampus_Measurements'
    query_image_path = input("Please provide query image filepath: ")
    retrain = True if input("Would you like to retrain the image AI? (y/n): ") == "y" else False
    height = float(input("Hippocampus height: "))
    width = float(input("Hippocampus width: "))
    whole_height = float(input("Whole brain height: "))
    weight = float(input("Weight multiplier for measurement differences: "))
    model_path = "triplet_model.pth"
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-3

    # === Prepare dataset and dataloader ===
    dataset = TripletDataset(reference_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # === Prepare measurement data ===
    PA2_Data = pd.read_csv(measurement_folder + "/PA2_Measurements.csv", index_col="Slice")
    PA4_Data = pd.read_csv(measurement_folder + "/PA4_Measurements.csv", index_col="Slice")
    PA5_Data = pd.read_csv(measurement_folder + "/PA5_Measurements.csv", index_col="Slice")

    # === Instantiate model, loss, optimizer ===
    model = ResNetEmbeddingNet().to(device)
    criterion = TripletLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # === Determine whether to retrain AI or use existing model ===
    if not retrain & os.path.exists(model_path): # use existing model instead of retraining
        print(f"Using existing model at {model_path}")

    else:  # retrain the AI
        # === Training loop ===
        for epoch in range(num_epochs):
            loss = train(model, dataloader, optimizer, criterion, device)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

        # === Save the trained model ===
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # === Load model for inference ===
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    reference_images = clean_file_list(sorted(os.listdir(reference_folder)))
    reference_paths = [os.path.join(reference_folder, f) for f in reference_images]

    reference_embeddings = []
    with torch.no_grad():
        for p in reference_paths:
            img_tensor = load_and_preprocess(p, transform)
            emb = model(img_tensor)
            reference_embeddings.append((p, emb.cpu().numpy()))

    # === Query image embedding ===
    query_img_tensor = load_and_preprocess(query_image_path, transform)
    with torch.no_grad():
        query_emb = model(query_img_tensor).cpu().numpy()

    # === Get a distance score for each reference based on measurements and image similarity ===
    distances = []
    for path, emb in reference_embeddings:
        dataset_num, slice_num = get_dataset_and_slice(path)
        dataset = PA2_Data if dataset_num == 2 else PA4_Data if dataset_num == 4 else PA5_Data

        try:
            curr_width = dataset.loc[slice_num, "Width"]
            curr_height = dataset.loc[slice_num, "Height"]
            curr_whole = dataset.loc[slice_num, "Whole_Height"]
        except Exception as e:
            print(e + f"\n WARNING: No slice data for slice {slice_num} from dataset PA{dataset_num}")
            continue

        measurement_dist = shape_difference((curr_width, curr_height, curr_whole), (width, height, whole_height)) * weight
        image_dist = euclidean_distance(query_emb, emb)
        dist = measurement_dist + image_dist
        print(f"slice {slice_num} of PA{dataset_num}: image dist = {image_dist}, measurement dist = {measurement_dist}")
        distances.append((path, dist))

    distances.sort(key=lambda x: x[1])
    top3 = distances[:3]

    # === Visualize top 3 matches ===
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    query_img = Image.open(query_image_path).convert("L")
    query_label = os.path.splitext(os.path.basename(query_image_path))[0]
    axes[0].imshow(query_img, cmap='gray')
    axes[0].set_title(f"Query:\n{query_label}")
    axes[0].axis('off')

    for i, (path, dist) in enumerate(top3):
        ref_img = Image.open(path).convert("L")
        axes[i+1].imshow(ref_img, cmap='gray')
        label = os.path.splitext(os.path.basename(path))[0]
        axes[i+1].set_title(f"{label}\nDist: {dist:.4f}")
        axes[i+1].axis('off')

    plt.show()

if __name__ == "__main__":
    main()
