# %%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore")
import os
import logging
import numpy as np
import scipy.io as sio
import h5py
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, cohen_kappa_score, precision_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from model.vision import DBCTNet, _3DRCNet, FAHM
from model.text import bert, t5, gpt
from model.base import Model
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# %%
class HyperspectralDataset(Dataset):
    def __init__(self, folder_name, patch_size=11):
        self.folder_name = folder_name
        self.patch_size = patch_size
        self.data, self.gt, self.captions, self.class_map = self.load_dataset()
        self.patches, self.labels, self.caption_texts, self.positions = self.create_patches()

    def load_dataset(self):
        data_file, gt_file, csv_file, class_map_file = None, None, None, None

        for file in os.listdir(self.folder_name):
            if file.endswith('_data.mat'):
                data_file = file
            elif file.endswith('_gt.mat'):
                gt_file = file
            elif file.endswith('.csv'):
                csv_file = file
            elif file.endswith('class_map.json'):
                class_map_file = file

        if not data_file or not gt_file:
            raise FileNotFoundError("Required .mat files not found in the folder.")

        try:
            data = sio.loadmat(os.path.join(self.folder_name, data_file))['data']
            gt = sio.loadmat(os.path.join(self.folder_name, gt_file))['gt']
        except:
            with h5py.File(os.path.join(self.folder_name, data_file), 'r') as f:
                data = np.array(f['data'])
            with h5py.File(os.path.join(self.folder_name, gt_file), 'r') as f:
                gt = np.array(f['gt'])

        captions = None
        if csv_file:
            captions = pd.read_csv(os.path.join(self.folder_name, csv_file))

        class_map = {}
        if class_map_file:
            with open(os.path.join(self.folder_name, class_map_file), 'r') as f:
                class_map = json.load(f)

        return data, gt, captions, class_map

    def create_patches(self):
        bands, height, width = self.data.shape
        pad = self.patch_size // 2
        padded_data = np.pad(self.data, ((0, 0), (pad, pad), (pad, pad)), mode='constant')

        patches_with_positions = []
        for i, j in product(range(height), range(width)):
            label = self.gt[i, j]
            if label == 0:
                continue

            patch = padded_data[:, i:i+self.patch_size, j:j+self.patch_size]
            patches_with_positions.append((patch, i, j))

        self.captions['Class name'] = self.captions['Class name'].str.strip().str.lower()
        class_caption_dict = {class_name: group['Description'].tolist() for class_name, group in self.captions.groupby('Class name')}

        patches, labels, captions, positions = [], [], [], []
        caption_indices = {class_name: 0 for class_name in class_caption_dict.keys()}

        print(f'Length of Patches: {len(patches_with_positions)}')
        for (patch, i, j) in patches_with_positions:
            label = self.gt[i, j] - 1
            class_name = self.class_map.get(str(int(label)), "Unknown").strip().lower()
            available_captions = class_caption_dict.get(class_name, ["No description available."])

            if caption_indices[class_name] >= len(available_captions):
                raise ValueError(f"Not enough unique captions for class {class_name}: {len(patches_with_positions)} samples, {len(available_captions)} captions.")

            caption = available_captions[caption_indices[class_name]]
            caption_indices[class_name] += 1

            patches.append(patch)
            labels.append(label)
            captions.append(caption)
            positions.append((i, j))

        return np.array(patches), np.array(labels), captions, positions

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return torch.tensor(self.patches[idx], dtype=torch.float32), self.labels[idx], self.caption_texts[idx], self.positions[idx]


dataset_folder='Datasets'
datasets = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
datasets.remove('KSC')

dataset=None
for dataset_name in datasets:
    print(f"Dataset: {dataset_name}")
    accuracies = []
    kappas = []
    dataset_path = os.path.join(dataset_folder, dataset_name)
    dataset = HyperspectralDataset(dataset_path, patch_size=11)
    # break

    # %%
    Output_dir = f'Caption_OUTPUTS/{dataset_name}'
    os.makedirs(Output_dir, exist_ok=True)

    # %%
    patch_size = dataset.patch_size
    num_classes = len(np.unique(dataset.labels))
    bands = dataset.patches.shape[1]

    # %%
    from datasets import Dataset
    def gen():
        for data in range(len(dataset)):
            image, _, caption, _ = dataset[data]
            yield {"image": image, "text": caption}
        
    ds = Dataset.from_generator(gen)
    split_ds = ds.train_test_split(test_size=0.2)

    ds = split_ds["train"].train_test_split(test_size=0.1)
    train_ds = split_ds["train"]
    test_ds = split_ds["test"]

    # %% [markdown]
    # ##### NEW TO MODIFY

    # %%
    from transformers import AutoProcessor
    from model.blipcaptioning import BlipForConditionalGeneration

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # %%
    from model.vision import FAHM
    import torch.nn as nn

    class VisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_vision = FAHM.FAHM(img_size=patch_size, in_chans=bands, n_groups=[2, 2, 2], depths=[1, 1, 1])
            self.fc = nn.Linear(64, 768)

        def forward(self, pixel_values,
                output_attentions = None,
                output_hidden_states= None,
                return_dict= None,
                interpolate_pos_encoding= None,):
            x = self.base_vision(pixel_values)
            x = self.fc(x)
            return [x.unsqueeze(1) ]

    model.vision_model = VisionModel().cuda()

    # %%
    import torch

    img = torch.randn(1,bands,patch_size,patch_size).cuda()
    model.vision_model(img)[0].shape

    # %%
    from torch.utils.data import Dataset, DataLoader

    class ImageCaptioningDataset(Dataset):
        def __init__(self, dataset, processor):
            self.dataset = dataset
            self.processor = processor

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            encoding = self.processor( text=item["text"], padding="max_length", return_tensors="pt")
            # remove batch dimension
            encoding = {k:v.squeeze() for k,v in encoding.items()}
            encoding["pixel_values"] = torch.tensor(item["image"], dtype=torch.float32)
            return encoding

    # %%
    train_dataset = ImageCaptioningDataset(train_ds, processor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)

    # %%
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()

    for epoch in range(50):
        print("Epoch:", epoch)
        loss_total = []
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)
            
            loss = outputs.loss

            loss_total.append(loss)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        print("Loss:", sum(loss_total)/len(loss_total))
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"{Output_dir}/model_{epoch}.pt")


