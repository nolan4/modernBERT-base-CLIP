import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from text_encoder import *
from vision_encoder import *
import os
import json
import numpy as np
import random
from tqdm import tqdm
import datetime

# Vision Caption Dataset
class VisionCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, captions_path, embeddings_dir, normalize=True):
        with open(captions_path, 'r') as f:
            self.captions_dict = json.load(f)

        self.embeddings_dir = embeddings_dir
        self.image_ids = list(self.captions_dict.keys())
        self.normalize = normalize

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        caption_entry = random.choice(self.captions_dict[image_id])
        tokenized_caption = caption_entry["tokenized"]
        attention_mask = caption_entry["attention_mask"]

        embedding_path = os.path.join(self.embeddings_dir, f"{image_id}.npy")
        embedding = np.load(embedding_path)

        embedding = torch.tensor(embedding, dtype=torch.float32)
        tokenized_caption = torch.tensor(tokenized_caption, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return embedding, tokenized_caption, attention_mask


class JointNetwork(nn.Module):
    def __init__(self):
        super(JointNetwork, self).__init__()
        
        self.text_encoder = modernBERT("answerdotai/ModernBERT-base")
        
        for param in self.text_encoder.parameters():
            param.requires_grad = True

        self.vision_projector = nn.Linear(1152, 512)
        self.text_projector = nn.Linear(768, 512)

    def forward(self, tokenized_text, image_encoding):
        vision_patch_pooled = image_encoding.mean(dim=1)  
        text_output = self.text_encoder(tokenized_text)  
        text_pooled = text_output.mean(dim=1)  

        vision_embedded = self.vision_projector(vision_patch_pooled)  
        text_embedded = self.text_projector(text_pooled)  

        vision_embedded = F.normalize(vision_embedded, dim=1)
        text_embedded = F.normalize(text_embedded, dim=1)

        return text_embedded, vision_embedded


def infoNCE_loss(text_features, vision_features, temperature=0.07):
    text_features = F.normalize(text_features, p=2, dim=-1)
    vision_features = F.normalize(vision_features, p=2, dim=-1)

    similarity_matrix = torch.matmul(text_features, vision_features.T) / temperature
    batch_size = vision_features.size(0)
    labels = torch.arange(batch_size, device=vision_features.device)

    loss_text_to_image = F.cross_entropy(similarity_matrix, labels)
    loss_image_to_text = F.cross_entropy(similarity_matrix.T, labels)

    return (loss_text_to_image + loss_image_to_text) / 2


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=5, freeze_text_encoder=True, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_loss = float('inf')  # Initialize with a very high value

    # Freeze text encoder if specified
    if freeze_text_encoder:
        for param in model.text_encoder.parameters():
            param.requires_grad = False

    # Ensure new layers are trainable
    for param in model.vision_projector.parameters():
        param.requires_grad = True
    for param in model.text_projector.parameters():
        param.requires_grad = True

    model.to(device)
    
    for epoch in range(num_epochs):

        # Train loop
        model.train()
        total_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{num_epochs} - Training:")
        train_progress = tqdm(train_loader, desc="Training", leave=True)

        for image_embeddings, tokenized_captions, attention_masks in train_progress:
            text_inputs = {"input_ids": tokenized_captions.to(device), "attention_mask": attention_masks.to(device)}
            image_embeddings = image_embeddings.to(device)

            optimizer.zero_grad()
            text_features, vision_features = model(text_inputs, image_embeddings)
            loss = infoNCE_loss(text_features, vision_features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())

        scheduler.step()

        # Validation Loop
        model.eval()
        val_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{num_epochs} - Validation:")
        val_progress = tqdm(val_loader, desc="Validation", leave=True)

        with torch.no_grad():
            for image_embeddings, tokenized_captions, attention_masks in val_progress:
                text_inputs = {"input_ids": tokenized_captions.to(device), "attention_mask": attention_masks.to(device)}
                image_embeddings = image_embeddings.to(device)

                text_features, vision_features = model(text_inputs, image_embeddings)
                loss = infoNCE_loss(text_features, vision_features)
                val_loss += loss.item()
                val_progress.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if checkpoint_path is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss
                }, checkpoint_path)
                print(f"New Best Model Saved at: {checkpoint_path} (Val Loss: {best_val_loss:.4f})")

    print("Training completed!")



if __name__ == "__main__":
    # Set random seed for reproducibility
    # torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths for dataset
    captions_path = '/mnt/nvme/shared_A/datasets/flickr30k/data/captions_tokenized.json'
    # embeddings_dir = '/mnt/nvme/shared_A/datasets/flickr30k/data/reduced_vision_embeddings'
    embeddings_dir = '/mnt/nvme/shared_A/datasets/flickr30k/data/vision_embeddings_reduced2'

    # Initialize datasets and loaders
    full_dataset = VisionCaptionDataset(captions_path, embeddings_dir)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

    # Initialize model, optimizer, and scheduler
    model = JointNetwork().to(device)

    checkpoint_path = f"./checkpoints/model_checkpoint_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"

    # **Phase 1 Configuration: Training new layers only**
    initial_lr = 1e-4
    min_lr = 1e-6
    num_epochs = 16
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

    # **Phase 1: Train new layers only, freeze text encoder**
    print("\n### Phase 1: Training new layers only (Text Encoder Frozen) ###")
    train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=num_epochs, freeze_text_encoder=True, checkpoint_path=checkpoint_path)

    # # **Phase 2 Configuration: Fine-tuning with adjusted learning rate**
    # initial_lr = 1e-4
    # min_lr = 1e-6
    # num_epochs = 3
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

    # print("\n### Phase 2: Fine-tuning text encoder and new layers ###")
    # train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=num_epochs, freeze_text_encoder=False, checkpoint_path=checkpoint_path)