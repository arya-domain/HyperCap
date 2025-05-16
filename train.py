import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore")
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
from model.vision import DBCTNet, _3DRCNet, FAHM, _3D_ConvSST
from model.text import bert, t5
from model.base import Model
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# Base directory for outputs
output_base_dir = None

# Create necessary directories
def setup_directories(dataset_name):
    dataset_dir = os.path.join(output_base_dir, dataset_name)
    logs_dir = os.path.join(dataset_dir, "logs")
    results_dir = os.path.join(dataset_dir, "results")
    maps_dir = os.path.join(dataset_dir, "maps")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)

    return logs_dir, results_dir, maps_dir

# Setup logging
def setup_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    while logger.handlers:
        handler = logger.handlers[0]
        handler.close()
        logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file, 'w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

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

def split_dataset(dataset, train_ratio=0.1, val_ratio=0.1, test_ratio=0.8):
    X_train, X_temp, y_train, y_temp, captions_train, captions_temp, positions_train, positions_temp = train_test_split(
        dataset.patches, dataset.labels, dataset.caption_texts, dataset.positions, test_size=1-train_ratio, stratify=dataset.labels, random_state=42
    )

    X_val, X_test, y_val, y_test, captions_val, captions_test, positions_val, positions_test = train_test_split(
        X_temp, y_temp, captions_temp, positions_temp, test_size=test_ratio/(test_ratio + val_ratio), stratify=y_temp, random_state=42
    )

    return (X_train, y_train, captions_train, positions_train), (X_val, y_val, captions_val, positions_val), (X_test, y_test, captions_test, positions_test)

def plot_classification_map(gt, pred_map, map_file):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(gt, cmap='jet', interpolation='nearest')
    axes[0].set_title("Ground Truth Classification Map")
    axes[0].axis("off")

    axes[1].imshow(pred_map, cmap='jet', interpolation='nearest')
    axes[1].set_title("Model-Predicted Classification Map")
    axes[1].axis("off")

    plt.savefig(map_file)
    plt.close(fig)

def train_model(model, train_loader, val_loader, epochs=20, lr=0.0001, logger=None):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float('inf')
    best_model_path = f"{output_base_dir}/best_model_.pth"

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0

        for x_batch, y_batch, caption, _ in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch, caption)
            loss = criterion(outputs, y_batch.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == y_batch).sum().item()

        train_acc = total_correct / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for x_val, y_val, caption, _ in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_outputs = model(x_val, caption)
                loss = criterion(val_outputs, y_val.long())
                val_loss += loss.item()
                val_correct += (val_outputs.argmax(dim=1) == y_val).sum().item()

        val_acc = val_correct / len(val_loader.dataset)
        val_loss /= len(val_loader)

        if logger:
            logger.info(f"Epoch {epoch+1}: Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            if logger:
                logger.info(f"Saved best model at epoch {epoch+1} with Val Loss: {val_loss:.4f}")

    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    return model

class HSI_ITER(Dataset):
    def __init__(self, patches, labels, captions, positions):
        self.patches = torch.tensor(patches, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.captions = captions
        self.positions = positions

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx], self.captions[idx], self.positions[idx]

def evaluate_model(model, test_loader, device):
    model.eval()
    model.to(device)

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for x_batch, y_batch, caption, _ in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch, caption)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Ensure we only access classes present in the report
    classes_in_report = [str(i) for i in range(1, len(precision) + 1) if str(i) in report]

    metrics_df = pd.DataFrame({
        "Class": classes_in_report,
        "Precision": [report[cls]['precision'] for cls in classes_in_report],
        "F1-Score": [report[cls]['f1-score'] for cls in classes_in_report],
        "Support": [report[cls]['support'] for cls in classes_in_report]
    })
    metrics_df.loc['Overall'] = ['Overall',  np.mean(precision), f1.mean(), len(y_true)]
    metrics_df.loc['Overall Kappa'] = ['Overall Kappa', kappa, 'N/A', 'N/A']
    metrics_df.loc['Overall Accuracy'] = ['Overall Accuracy', accuracy, 'N/A', 'N/A']
    metrics_df.loc['Overall Precision'] = ['Overall Precision',  np.mean(precision), 'N/A', 'N/A']
    metrics_df.loc['Overall F1-Score'] = ['Overall F1-Score',  np.mean(f1), 'N/A', 'N/A']

    return metrics_df, accuracy, kappa

def predict_dataset_for_map(model, dataset, device):
    model.eval()
    model.to(device)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions = []
    positions = []
    with torch.no_grad():
        for batch in data_loader:
            patch, _, caption, pos = batch
            patch = patch.to(device)
            outputs = model(patch, caption)
            preds = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds + 1)
            positions.extend([pos])

    pred_map = np.zeros_like(dataset.gt)
    for (pred, (i, j)) in zip(predictions, positions):
        if dataset.gt[i, j] != 0:
            pred_map[i, j] = pred

    return pred_map

def run_pipeline(dataset_folder, merging_methods, num_runs=3, epochs=30, batch_size=32, use_only=None):
    datasets = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]

    for dataset_name in datasets:
        accuracies = []
        kappas = []
        dataset_path = os.path.join(dataset_folder, dataset_name)
        dataset = HyperspectralDataset(dataset_path, patch_size=11)
        bands, height, width = dataset.data.shape
        patch_size = dataset.patch_size
        
        VISION_MODEL_INFO ={
            'DBCTNet' : {
                'out_dim': 16,
                'model': DBCTNet.DBCTNet(bands=bands, patch=patch_size),
            },
            '3DRCNet' : {
                'out_dim': 256,
                'model': _3DRCNet.ConvNeXt(depths=[1, 2, 4, 2], dims=[32, 64, 128, 256]),
            },
            'FAHM' : {
                'out_dim': 64,
                'model': FAHM.FAHM(img_size=patch_size, in_chans=bands, n_groups=[2, 2, 2], depths=[1, 1, 1]),
            },
            '3D_ConvSST' : {
                'out_dim': 64,
                'model': _3D_ConvSST.VisionTransformer(bands, embed_dim=64, depth=2, n_heads=8),
            },
            'None' : {
                'out_dim': 512,
                'model': None,
            },
        }
        TEXT_MODEL_INFO ={
            'BertEnocder_Large' : {
                'out_dim': 1024,
                'model': bert.BertEnocder(),
            },
            'T5Encoder_Large' : {
                'out_dim': 1024,
                'model': t5.T5Encoder(),
            },
            'None' : {
                'out_dim': 512,
                'model': None,
            },
        }

        if use_only == None:
            vision_keys = list(VISION_MODEL_INFO.keys())
            vision_keys.remove('None')
            text_keys = list(TEXT_MODEL_INFO.keys())
            text_keys.remove('None')
            MODEL_COMBINATIONS = list(product(vision_keys, text_keys))
        elif use_only == 'vision':
            vision_keys = list(VISION_MODEL_INFO.keys())
            vision_keys.remove('None')
            MODEL_COMBINATIONS = list(product(vision_keys, ['None']))
            merging_methods = ['vision']
        else:
            text_keys = list(TEXT_MODEL_INFO.keys())
            text_keys.remove('None')
            MODEL_COMBINATIONS = list(product(['None'], text_keys))
            merging_methods = ['text']
        
        (train_patches, train_labels,
        train_captions, train_positions), (val_patches,
        val_labels, val_captions, val_positions
        ), (test_patches, test_labels,
        test_captions, test_positions) = split_dataset(
                        dataset,
                        train_ratio=0.1,
                        val_ratio=0.1,
                        test_ratio=0.8
                        )

        train_dataset = HSI_ITER(train_patches, train_labels, train_captions, train_positions)
        val_dataset = HSI_ITER(val_patches, val_labels, val_captions, val_positions)
        test_dataset = HSI_ITER(test_patches, test_labels, test_captions, test_positions)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for vision_encoder_name, text_encoder_name in tqdm(MODEL_COMBINATIONS, desc="Processing combinations"):
            logs_dir, results_dir, maps_dir = setup_directories(os.path.join(dataset_name, f'{vision_encoder_name}-{text_encoder_name}'))
            for merging_method in tqdm(merging_methods, desc="Processing merging methods"):
                for run in range(1, num_runs + 1): 
                    log_file = os.path.join(logs_dir, f"run{run}_{merging_method}.log")
                    logger = setup_logging(log_file)
                    logger.info(f"Starting run {run} with {merging_method} merging method Vision ENCODER: {vision_encoder_name} Text ENCODER: {text_encoder_name}")

                    vision_encoder = VISION_MODEL_INFO[vision_encoder_name]['model']
                    vision_out = VISION_MODEL_INFO[vision_encoder_name]['out_dim']

                    text_encoder = TEXT_MODEL_INFO[text_encoder_name]['model']
                    text_out = TEXT_MODEL_INFO[text_encoder_name]['out_dim']

                    num_classes = len(np.unique(train_labels))
                    model = Model(num_classes,
                                vision_encoder,
                                text_encoder,
                                vision_out,
                                text_out,
                                merging_method=merging_method,
                                use_only=use_only
                                )

                    model = train_model(model, train_loader, val_loader, epochs=epochs, lr=0.0001, logger=logger)

                    results_file = os.path.join(results_dir, f"{merging_method}_results_run{run}.csv")
                    metrics_df, accuracy, kappa = evaluate_model(model, test_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                    metrics_df.to_csv(results_file, index=False)

                    pred_map = predict_dataset_for_map(model, dataset, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                    map_file = os.path.join(maps_dir, f"{merging_method}_prediction_map_run{run}.png")
                    plot_classification_map(dataset.gt, pred_map, map_file)

                    accuracies.append(accuracy)
                    kappas.append(kappa)
                    logger.info(f"Run {run} completed with Accuracy: {accuracy:.4f}, Kappa: {kappa:.4f}")

                summary_results = {
                    "Merging Method": [merging_method],
                    "Mean Accuracy": [np.mean(accuracies)],
                    "Std Accuracy": [np.std(accuracies)],
                    "Mean Kappa": [np.mean(kappas)],
                    "Std Kappa": [np.std(kappas)],
                    "Average Accuracy": [np.mean(accuracies)]
                }
                summary_df = pd.DataFrame(summary_results)
                summary_df.to_csv(os.path.join(results_dir, f"{merging_method}_summary_results.csv"), index=False)

# Example usage
dataset_folder = "./Datasets" # Datasets
merging_methods = ['CONCAT', 'PWA', 'PWM', 'MHA', 'CA'] 

# Base directory for outputs 
output_base_dir = "OUTPUTS/"

# use only ---> 'vision' 'text' None
run_pipeline(dataset_folder, merging_methods, num_runs=1, epochs=50, batch_size=32, use_only=None)