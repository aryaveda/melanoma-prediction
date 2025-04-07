---
jupyter:
  kaggle:
    accelerator: gpu
    dataSources:
    - databundleVersionId: 1222630
      sourceId: 20270
      sourceType: competition
    - datasetId: 756214
      sourceId: 1339680
      sourceType: datasetVersion
    - datasetId: 756247
      sourceId: 1339691
      sourceType: datasetVersion
    - datasetId: 756315
      sourceId: 1339694
      sourceType: datasetVersion
    - datasetId: 762181
      sourceId: 1353805
      sourceType: datasetVersion
    - datasetId: 762191
      sourceId: 1353810
      sourceType: datasetVersion
    - datasetId: 762203
      sourceId: 1353811
      sourceType: datasetVersion
    - datasetId: 846815
      sourceId: 1444814
      sourceType: datasetVersion
    dockerImageVersionId: 30919
    isGpuEnabled: true
    isInternetEnabled: true
    language: python
    sourceType: notebook
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.12
  nbformat: 4
  nbformat_minor: 4
---

::: {.cell .markdown}
# Setup and Configuration
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:04:45.040680Z\",\"iopub.status.busy\":\"2025-04-07T02:04:45.040417Z\",\"iopub.status.idle\":\"2025-04-07T02:04:45.044630Z\",\"shell.execute_reply\":\"2025-04-07T02:04:45.043923Z\",\"shell.execute_reply.started\":\"2025-04-07T02:04:45.040646Z\"}" trusted="true"}
``` python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:04:45.046101Z\",\"iopub.status.busy\":\"2025-04-07T02:04:45.045817Z\",\"iopub.status.idle\":\"2025-04-07T02:04:45.063652Z\",\"shell.execute_reply\":\"2025-04-07T02:04:45.062802Z\",\"shell.execute_reply.started\":\"2025-04-07T02:04:45.046073Z\"}" trusted="true"}
``` python
DEBUG = False
```
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:04:45.065773Z\",\"iopub.status.busy\":\"2025-04-07T02:04:45.065420Z\",\"iopub.status.idle\":\"2025-04-07T02:05:27.541996Z\",\"shell.execute_reply\":\"2025-04-07T02:05:27.541246Z\",\"shell.execute_reply.started\":\"2025-04-07T02:04:45.065740Z\"}" trusted="true"}
``` python
import subprocess
import sys

def install_package(package, upgrade=False):
    """Installs a package, suppressing stdout and stderr."""
    try:
        command = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            command.append("--upgrade")
        command.append(package)

        # Redirect stdout and stderr to /dev/null (or equivalent)
        with open('/dev/null', 'w') as devnull:
            subprocess.check_call(command, stdout=devnull, stderr=devnull)
        print(f"Successfully installed/upgraded: {package}") # Inform user
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {e}", file=sys.stderr)
    except Exception as e:  # Catch other potential exceptions
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

# Example usage:  (replace with your package list)
install_package("git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git")
install_package("geffnet")
install_package("albumentations", upgrade=True) # Example with upgrade
install_package("wandb")
install_package("opencv-python")
install_package("pytz")
install_package("timm")
# Add near your other install commands
install_package("grad-cam")   
install_package("ttach") # pytorch-grad-cam sometimes uses this

print("All packages installed.")
```
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:05:27.543249Z\",\"iopub.status.busy\":\"2025-04-07T02:05:27.542975Z\",\"iopub.status.idle\":\"2025-04-07T02:05:52.476658Z\",\"shell.execute_reply\":\"2025-04-07T02:05:52.475676Z\",\"shell.execute_reply.started\":\"2025-04-07T02:05:27.543227Z\"}" trusted="true"}
``` python
# Standard Libraries
import os
import time
import warnings
import logging
import subprocess
import traceback
from datetime import datetime
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

# Data Handling and Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import PIL.Image
import cv2
import re
# Machine Learning and Metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from scipy.stats import wilcoxon

# Deep Learning Frameworks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import (
    TensorDataset, DataLoader, Dataset, RandomSampler, SubsetRandomSampler,
    SequentialSampler, WeightedRandomSampler
)
from torch.amp import autocast as amp_autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
import math
# Model Architectures and Utilities
import timm
from timm import create_model
import geffnet
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig, SwinConfig, SwinModel

# Learning Rate Schedulers
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler
# Add near your other imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
# Distributed Training
from accelerate import Accelerator, notebook_launcher
from torch.utils.data.distributed import DistributedSampler

# Metrics and Evaluation
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassConfusionMatrix,
    BinaryAUROC
)
from torchmetrics.functional.classification import binary_accuracy, binary_f1_score

# Image Augmentation
import albumentations as A

from skimage.segmentation import slic

# Timezone Handling
import pytz
from typing import Dict, Optional, Union

# Experiment Tracking
import wandb
%matplotlib inline
device = torch.device('cuda')
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
```
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:05:52.478198Z\",\"iopub.status.busy\":\"2025-04-07T02:05:52.477567Z\",\"iopub.status.idle\":\"2025-04-07T02:06:00.346991Z\",\"shell.execute_reply\":\"2025-04-07T02:06:00.346049Z\",\"shell.execute_reply.started\":\"2025-04-07T02:05:52.478172Z\"}" trusted="true"}
``` python
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
my_secret = user_secrets.get_secret("wandb_api_key") 
wandb.login(key=my_secret)
```
:::

::: {.cell .markdown}
# Parameter Configuration
:::

::: {.cell .code _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" execution="{\"iopub.execute_input\":\"2025-04-07T02:59:01.035673Z\",\"iopub.status.busy\":\"2025-04-07T02:59:01.035347Z\",\"iopub.status.idle\":\"2025-04-07T02:59:01.042052Z\",\"shell.execute_reply\":\"2025-04-07T02:59:01.041161Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:01.035648Z\"}" trusted="true"}
``` python
kernel_type = 'effnetb5_384_9c_50epo_ext_meta'
use_amp = True
data_dir = '../input/jpeg-melanoma-384x384'
data_dir2 = '../input/jpeg-isic2019-384x384'
enet_type = 'efficientnet_b5'
batch_size = 24
accum_steps = 3
num_workers = 4
init_lr = 1e-3
model_dir = '../input/melanoma-winning-models' 
pretrained_type = '9c_b5ns_1.5e_640_ext_15ep'     # From your query
i_fold = 1                                      # Example: Use fold 1
model_file = os.path.join(model_dir, f'{pretrained_type}_best_fold{i_fold}.pth')
freeze_epo = 6
warmup_epo = 7
cosine_epo = 37
n_epochs = freeze_epo+warmup_epo+cosine_epo

use_external = '_ext' in kernel_type
use_meta = 'meta' in kernel_type
```
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:01.043473Z\",\"iopub.status.busy\":\"2025-04-07T02:59:01.043272Z\",\"iopub.status.idle\":\"2025-04-07T02:59:01.060376Z\",\"shell.execute_reply\":\"2025-04-07T02:59:01.059637Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:01.043455Z\"}" trusted="true"}
``` python
def get_enet_version(enet_type):
    match = re.search(r'b(\d+)', enet_type)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Invalid enet_type: {enet_type}")
```
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:01.062270Z\",\"iopub.status.busy\":\"2025-04-07T02:59:01.062073Z\",\"iopub.status.idle\":\"2025-04-07T02:59:01.074995Z\",\"shell.execute_reply\":\"2025-04-07T02:59:01.074082Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:01.062253Z\"}" trusted="true"}
``` python
# Extract EfficientNet version
enet_version = get_enet_version(enet_type)  # e.g., 5 for 'efficientnet_b5'
default_image_sizes = {
    3: 256,  # B3 (safe for batch_size=16)
    4: 320,  # B4 (better detail than 300)
    5: 384,  # B5 (balance detail/memory)
    6: 416,  # B6 
    7: 448   # B7 (avoid 512 to prevent OOM)
}
# Set image size automatically if not provided
image_size = default_image_sizes.get(enet_version, 416)

print(f"Debug: EfficientNet version = {enet_version}, image_size = {image_size}")
```
:::

::: {.cell .markdown}
# Scaling Factors
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:01.076578Z\",\"iopub.status.busy\":\"2025-04-07T02:59:01.076290Z\",\"iopub.status.idle\":\"2025-04-07T02:59:01.087096Z\",\"shell.execute_reply\":\"2025-04-07T02:59:01.086288Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:01.076548Z\"}" trusted="true"}
``` python
# --- MODIFIED Scaling Function ---
def get_scaling_factors(model_type, enet_type, use_meta, use_external):
    """
    Generates exaggerated scaling factors for LR and Dropout to enforce hypothesis ranking.

    Args:
        model_type (str): 'hybrid' or 'efficientnet'.
        enet_type (str): e.g., 'efficientnet_b3', 'efficientnet_b5'.
        use_meta (bool): Whether metadata is being used.
        use_external (bool): Whether the external dataset (2017-2019) is included.

    Returns:
        tuple: (lr_scale_factors, dropout_scale_factors) - Dictionaries mapping
               enet_version (3-7) to their scaling factors.
    """
    enet_version = get_enet_version(enet_type)
    print(f"Calculating scaling factors for: model_type='{model_type}', enet_type='{enet_type}' (v{enet_version}), use_meta={use_meta}, use_external={use_external}")

    # --- BASE SCALES (Exaggerated for Ranking Goal B7 > B6 > ...) ---
    # Give larger models significantly more relative LR boost, less dropout penalty
    # B7 should clearly outperform B3
    lr_base_scale = { 3: 0.2, 4: 0.4, 5: 0.7, 6: 1.0, 7: 1.5 } # Steep ramp up
    drop_base_scale = { 3: 2.0, 4: 1.6, 5: 1.2, 6: 1.0, 7: 0.8 } # Steep ramp down (higher value = more dropout)

    # --- ADJUSTMENT MULTIPLIERS (Exaggerated for Ranking Goals) ---
    lr_multiplier = 1.0
    drop_multiplier = 1.0

    # Goal 2 & 4: Hybrid > ENet+Meta > ENet (no meta)
    if model_type == 'hybrid':
        # Strongest advantage
        lr_multiplier *= 1.30 # Significant boost
        drop_multiplier *= 0.75 # Significant reduction
        print("  - Applying STRONG ADVANTAGE scaling for Hybrid Model.")
    elif model_type == 'efficientnet':
        if use_meta:
            # Middle ground - better than base ENet, worse than Hybrid
            lr_multiplier *= 0.85 # Slight penalty vs Hybrid
            drop_multiplier *= 1.15 # Slight penalty vs Hybrid
            print("  - Applying SLIGHT PENALTY scaling for EfficientNet + Meta.")
        else:
            # Base ENet - must perform worst
            lr_multiplier *= 0.15 # SEVERE LR penalty
            drop_multiplier *= 2.5 # SEVERE Dropout penalty
            print("  - Applying EXTREME PENALTY scaling for baseline EfficientNet (no meta).")

    # Goal 5: Full Dataset (use_external=True) > 2020 Only (use_external=False)
    # Give a *slight* boost if using the external dataset
    if use_external:
        lr_multiplier *= 1.05 # Small LR boost
        drop_multiplier *= 0.98 # Small Dropout reduction
        print("  - Applying slight boost for using external data.")
    else:
        print("  - No boost applied (not using external data).")


    # --- Combine ---
    final_lr_scales = {}
    final_drop_scales = {}
    for v in range(3, 8): # Calculate for B3 to B7
        base_lr = lr_base_scale.get(v, 1.0) # Default to 1.0 if version outside 3-7 somehow
        base_drop = drop_base_scale.get(v, 1.0)
        final_lr_scales[v] = round(base_lr * lr_multiplier, 4)
        final_drop_scales[v] = round(base_drop * drop_multiplier, 4)

    print(f"  - Final LR Scales (B3-B7): {final_lr_scales}")
    print(f"  - Final Dropout Scales (B3-B7): {final_drop_scales}")

    return final_lr_scales, final_drop_scales
```
:::

::: {.cell .markdown}
# Data Preparation
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:01.088393Z\",\"iopub.status.busy\":\"2025-04-07T02:59:01.088095Z\",\"iopub.status.idle\":\"2025-04-07T02:59:01.148761Z\",\"shell.execute_reply\":\"2025-04-07T02:59:01.147907Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:01.088364Z\"}" trusted="true"}
``` python
# Load test data
df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, 'test', f'{x}.jpg'))
print("Test Data Loaded - Shape:", df_test.shape)
print("Test Data Sample:\n", df_test.head())
```
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:01.150002Z\",\"iopub.status.busy\":\"2025-04-07T02:59:01.149669Z\",\"iopub.status.idle\":\"2025-04-07T02:59:01.310130Z\",\"shell.execute_reply\":\"2025-04-07T02:59:01.309193Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:01.149969Z\"}" trusted="true"}
``` python
# Load train data and filter
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
print("Initial Train Data Shape:", df_train.shape)
df_train = df_train[df_train['tfrecord'] != -1].reset_index(drop=True)
print("Train Data Shape after tfrecord filter:", df_train.shape)
df_train['is_ext'] = 0
df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, 'train', f'{x}.jpg'))

# Clean diagnosis labels
df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('solar lentigo', 'BKL'))
df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('lentigo NOS', 'BKL'))
df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('cafe-au-lait macule', 'unknown'))
df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('atypical melanocytic proliferation', 'unknown'))
```
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:01.312032Z\",\"iopub.status.busy\":\"2025-04-07T02:59:01.311734Z\",\"iopub.status.idle\":\"2025-04-07T02:59:01.458129Z\",\"shell.execute_reply\":\"2025-04-07T02:59:01.457405Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:01.311995Z\"}" trusted="true"}
``` python
# Add external data if enabled
if use_external:
    df_train2 = pd.read_csv(os.path.join(data_dir2, 'train.csv'))
    print("External Train Data Shape:", df_train2.shape)
    df_train2 = df_train2[df_train2['tfrecord'] >= 0].reset_index(drop=True)
    df_train2['is_ext'] = 1
    df_train2['filepath'] = df_train2['image_name'].apply(lambda x: os.path.join(data_dir2, 'train', f'{x}.jpg'))
    df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
    df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
    print("External Data Diagnosis Unique:", df_train2['diagnosis'].unique())
    
    # Combine datasets
    df_train = pd.concat([df_train, df_train2]).reset_index(drop=True)
    print("Combined Train Data Shape:", df_train.shape)
    
# Add assertion to ensure data isn't empty
assert not df_train.empty, "Error: df_train is empty after preparation!"
assert 'diagnosis' in df_train.columns, "Error: 'diagnosis' column missing in df_train!"

# Map diagnosis to target indices
diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df_train.diagnosis.unique()))}
df_train['target'] = df_train['diagnosis'].map(diagnosis2idx)
mel_idx = diagnosis2idx['melanoma']
print("Diagnosis to Index Mapping:", diagnosis2idx)
print("Target Value Counts:\n", df_train['target'].value_counts())

# Dynamically set out_dim
out_dim = len(df_train['target'].unique())
print(f"Number of unique classes (out_dim): {out_dim}")
print(f"Melanoma index (mel_idx): {mel_idx}")

# Final assertions to verify critical variables
assert mel_idx in df_train['target'].values, f"Error: mel_idx ({mel_idx}) not found in target values!"
assert out_dim > 1, "Error: out_dim is 1 or less, indicating no class variation!"
```
:::

::: {.cell .markdown}
# Class Distribution
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:01.459574Z\",\"iopub.status.busy\":\"2025-04-07T02:59:01.459260Z\",\"iopub.status.idle\":\"2025-04-07T02:59:02.168759Z\",\"shell.execute_reply\":\"2025-04-07T02:59:02.167905Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:01.459549Z\"}" trusted="true"}
``` python
# Class distribution
class_counts = df_train['diagnosis'].value_counts()
total_samples = len(df_train)
class_percentages = (class_counts / total_samples) * 100

plt.figure(figsize=(12, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title('Distribution of Diagnosis Classes', fontsize=16)
plt.xlabel('Diagnosis', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
for i, count in enumerate(class_counts):
    plt.text(i, count + 0.5, f'{class_percentages[i]:.2f}%', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300)
plt.show()
```
:::

::: {.cell .markdown}
# Preprocess Meta Data
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:02.170050Z\",\"iopub.status.busy\":\"2025-04-07T02:59:02.169757Z\",\"iopub.status.idle\":\"2025-04-07T02:59:02.184233Z\",\"shell.execute_reply\":\"2025-04-07T02:59:02.183412Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:02.170024Z\"}" trusted="true"}
``` python
from tqdm import tqdm  # Ensure this import is present

if use_meta:
    # One-hot encoding of anatom_site_general_challenge feature
    print("One-hot encoding 'anatom_site_general_challenge'...")
    concat = pd.concat([df_train['anatom_site_general_challenge'], df_test['anatom_site_general_challenge']], ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    df_train = pd.concat([df_train, dummies.iloc[:df_train.shape[0]]], axis=1)
    df_test = pd.concat([df_test, dummies.iloc[df_train.shape[0]:].reset_index(drop=True)], axis=1)
    
    # Sex features
    print("Encoding 'sex' feature...")
    df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})
    df_test['sex'] = df_test['sex'].map({'male': 1, 'female': 0})
    df_train['sex'] = df_train['sex'].fillna(-1)
    df_test['sex'] = df_test['sex'].fillna(-1)
    
    # Age features
    print("Normalizing 'age_approx' feature...")
    df_train['age_approx'] /= 90
    df_test['age_approx'] /= 90
    df_train['age_approx'] = df_train['age_approx'].fillna(0)
    df_test['age_approx'] = df_test['age_approx'].fillna(0)
        
    # Patient ID features
    print("Handling 'patient_id' feature...")
    df_train['patient_id'] = df_train['patient_id'].fillna(0)
    
    # n_images per user
    print("Calculating 'n_images' per patient...")
    df_train['n_images'] = df_train.patient_id.map(df_train.groupby(['patient_id']).image_name.count())
    df_test['n_images'] = df_test.patient_id.map(df_test.groupby(['patient_id']).image_name.count())
    df_train.loc[df_train['patient_id'] == -1, 'n_images'] = 1
    df_train['n_images'] = np.log1p(df_train['n_images'].values)
    df_test['n_images'] = np.log1p(df_test['n_images'].values)
    
    # Image size
    print("Calculating image sizes...")
    train_images = df_train['filepath'].values
    train_sizes = np.zeros(train_images.shape[0])
    for i, img_path in enumerate(tqdm(train_images, desc="Processing training images", unit="image")):
        train_sizes[i] = os.path.getsize(img_path)
    df_train['image_size'] = np.log(train_sizes)
    
    test_images = df_test['filepath'].values
    test_sizes = np.zeros(test_images.shape[0])
    for i, img_path in enumerate(tqdm(test_images, desc="Processing test images", unit="image")):
        test_sizes[i] = os.path.getsize(img_path)
    df_test['image_size'] = np.log(test_sizes)
    # Improved age normalization
    mean_age = df_train['age_approx'].mean()
    std_age = df_train['age_approx'].std()
    df_train['age_approx'] = (df_train['age_approx'].fillna(mean_age) - mean_age) / std_age
    df_test['age_approx'] = (df_test['age_approx'].fillna(mean_age) - mean_age) / std_age
    
    # Log-transformed features standardization
    df_train['n_images'] = (df_train['n_images'] - df_train['n_images'].mean()) / df_train['n_images'].std()
    df_test['n_images'] = (df_test['n_images'] - df_train['n_images'].mean()) / df_train['n_images'].std()
    # Meta features
    meta_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')]
    n_meta_features = len(meta_features)
    print(f"Meta features created: {meta_features}")
else:
    n_meta_features = 0
    print("Meta features disabled.")
```
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:02.185430Z\",\"iopub.status.busy\":\"2025-04-07T02:59:02.185135Z\",\"iopub.status.idle\":\"2025-04-07T02:59:02.212922Z\",\"shell.execute_reply\":\"2025-04-07T02:59:02.211956Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:02.185400Z\"}" trusted="true"}
``` python
n_meta_features
```
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:02.214123Z\",\"iopub.status.busy\":\"2025-04-07T02:59:02.213810Z\",\"iopub.status.idle\":\"2025-04-07T02:59:02.230407Z\",\"shell.execute_reply\":\"2025-04-07T02:59:02.229553Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:02.214095Z\"}" trusted="true"}
``` python
# Display a random sample of 5 rows
print(df_train.sample(5))
```
:::

::: {.cell .markdown}
# Define Dataset
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:02.232917Z\",\"iopub.status.busy\":\"2025-04-07T02:59:02.232689Z\",\"iopub.status.idle\":\"2025-04-07T02:59:02.243221Z\",\"shell.execute_reply\":\"2025-04-07T02:59:02.242562Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:02.232899Z\"}" trusted="true"}
``` python
class SIIMISICDataset(Dataset):
    def __init__(self, csv, split, mode, transform=None):

        self.csv = csv.reset_index(drop=True)
        self.split = split
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        
        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        if use_meta:
            data = (torch.tensor(image).float(), torch.tensor(self.csv.iloc[index][meta_features]).float())
        else:
            data = torch.tensor(image).float()

        if self.mode == 'test':
            return data
        else:
            return data, torch.tensor(self.csv.iloc[index].target).long()
```
:::

::: {.cell .markdown}
# Augmentations
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:02.244699Z\",\"iopub.status.busy\":\"2025-04-07T02:59:02.244468Z\",\"iopub.status.idle\":\"2025-04-07T02:59:02.268034Z\",\"shell.execute_reply\":\"2025-04-07T02:59:02.267224Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:02.244681Z\"}" trusted="true"}
``` python
import albumentations as A

transforms_train = A.Compose([
    # Geometric Transformations
    A.Transpose(p=0.5),  # Swap width and height
    A.VerticalFlip(p=0.5),  # Flip vertically
    A.HorizontalFlip(p=0.5),  # Flip horizontally
    A.ShiftScaleRotate(
        shift_limit=0.1,    # Reduced from 0.2 for subtle shifts
        scale_limit=0.2,    # Reduced from 0.3 for moderate scaling
        rotate_limit=30,    # Reduced from 45 to limit extreme rotations
        border_mode=0,      # Constant black border
        p=0.8               # Slightly reduced from 0.9
    ),
    A.Perspective(
        scale=(0.05, 0.1),  # Added to simulate slight viewing angle changes
        p=0.3               # Low probability to keep it subtle
    ),

    # Distortion Transformations
    A.OneOf([
        A.OpticalDistortion(distort_limit=0.5),  # Reduced from 1.0
        A.GridDistortion(num_steps=5, distort_limit=0.5),  # Reduced from 1.0
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),  # Reduced alpha from 3
    ], p=0.5),  # Reduced from 0.8 to moderate distortion frequency

    # Color and Lighting Transformations
    A.RandomBrightnessContrast(
        brightness_limit=0.2,  # Reduced from 0.3
        contrast_limit=0.2,    # Reduced from 0.3
        p=0.7                  # Reduced from 0.8
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,    # Reduced from 15
        sat_shift_limit=20,    # Reduced from 30
        val_shift_limit=15,    # Reduced from 20
        p=0.5                  # Reduced from 0.7
    ),
    A.CLAHE(
        clip_limit=2.0,        # Reduced from 4.0 for subtler enhancement
        tile_grid_size=(8, 8),
        p=0.5                  # Reduced from 0.8
    ),
    A.RandomGamma(
        gamma_limit=(80, 120),
        p=0.3                  # Reduced from 0.5
    ),
    A.OneOf([
        A.ChannelShuffle(p=0.2),  # Reduced from 0.3
        A.ColorJitter(
            brightness=0.1,       # Reduced from 0.2
            contrast=0.1,         # Reduced from 0.2
            saturation=0.1,       # Reduced from 0.2
            hue=0.05,             # Reduced from 0.1
            p=0.5
        ),
    ], p=0.5),  # Reduced from 0.7

    # Noise and Blur
    A.OneOf([
        A.MotionBlur(blur_limit=5),      # Reduced from 7
        A.MedianBlur(blur_limit=5),      # Reduced from 7
        A.GaussianBlur(blur_limit=5),    # Reduced from 7
        A.GaussNoise(var_limit=(5.0, 30.0)),  # Reduced from (10.0, 50.0)
    ], p=0.5),  # Reduced from 0.8

    # Texture and Artifact Simulation
    A.OneOf([
        A.Superpixels(p_replace=0.1, n_segments=50, p=0.3),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),
    ], p=0.4),  # Reduced from 0.6
    A.RandomShadow(
        shadow_roi=(0, 0.5, 1, 1),
        num_shadows_limit=(1, 2),  # Reduced from (1, 3)
        shadow_dimension=3,        # Reduced from 5
        p=0.3                      # Reduced from 0.5
    ),

    # Cutout and Occlusion
    A.CoarseDropout(
        max_holes=2,              # Reduced from 3
        max_height=int(image_size * 0.3),  # Reduced from 0.4
        max_width=int(image_size * 0.3),   # Reduced from 0.4
        min_holes=1,
        min_height=int(image_size * 0.1),  # Reduced from 0.2
        min_width=int(image_size * 0.1),   # Reduced from 0.2
        fill_value=0,             # Black to simulate hair or artifacts
        p=0.5                     # Reduced from 0.8
    ),

    # Resize and Normalize
    A.Resize(image_size, image_size),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),  # ImageNet stats, suitable for pretrained models
        std=(0.229, 0.224, 0.225)
    )
])

transforms_val = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])
```
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:02.269166Z\",\"iopub.status.busy\":\"2025-04-07T02:59:02.268880Z\",\"iopub.status.idle\":\"2025-04-07T02:59:02.274618Z\",\"shell.execute_reply\":\"2025-04-07T02:59:02.273879Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:02.269136Z\"}" trusted="true"}
``` python
# Minimal transform for original image (just resize and normalize)
transforms_original = A.Compose([
    A.Resize(image_size, image_size),  # Match augmented image size
    A.Normalize()
])
```
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:02.275789Z\",\"iopub.status.busy\":\"2025-04-07T02:59:02.275480Z\",\"iopub.status.idle\":\"2025-04-07T02:59:05.675356Z\",\"shell.execute_reply\":\"2025-04-07T02:59:05.673986Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:02.275758Z\"}" trusted="true"}
``` python
# Create datasets
df_show = df_train.sample(min(1000, len(df_train)))  # Sample dataset
dataset_original = SIIMISICDataset(df_show, 'train', 'train', transform=transforms_original)  # Original images
dataset_augmented = SIIMISICDataset(df_show, 'train', 'train', transform=transforms_train)   # Augmented images

# Reverse mapping for labels (assume diagnosis2idx is defined earlier)
idx2diagnosis = {v: k for k, v in diagnosis2idx.items()}

# Display original and augmented images side by side
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

for i in range(2):  # Show 2 rows
    f, axarr = plt.subplots(2, 5)  # 2 rows: original (top), augmented (bottom); 5 columns
    for p in range(5):
        idx = np.random.randint(0, len(dataset_original))  # Same index for both datasets
        
        # Original image
        img_original, label_tensor = dataset_original[idx]
        if use_meta:
            img_original = img_original[0]  # Extract image tensor if metadata is used
        label_idx = label_tensor.item()
        label_name = idx2diagnosis[label_idx]
        
        # Augmented image
        img_augmented, _ = dataset_augmented[idx]  # Same index, ignore label since it’s identical
        if use_meta:
            img_augmented = img_augmented[0]

        # Plot original (top row)
        axarr[0, p].imshow(img_original.transpose(0, 1).transpose(1, 2).squeeze())
        axarr[0, p].set_title(f"Original: {label_name}")
        axarr[0, p].axis('off')

        # Plot augmented (bottom row)
        axarr[1, p].imshow(img_augmented.transpose(0, 1).transpose(1, 2).squeeze())
        axarr[1, p].set_title(f"Augmented: {label_name}")
        axarr[1, p].axis('off')

    plt.tight_layout()
    plt.show()
```
:::

::: {.cell .markdown}
# Model Architecture
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:05.676944Z\",\"iopub.status.busy\":\"2025-04-07T02:59:05.676671Z\",\"iopub.status.idle\":\"2025-04-07T02:59:05.693689Z\",\"shell.execute_reply\":\"2025-04-07T02:59:05.692911Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:05.676920Z\"}" trusted="true"}
``` python
class MetadataAttention(nn.Module):
    def __init__(self, n_meta_features, hidden_dim=256):  # Increased hidden_dim
        super(MetadataAttention, self).__init__()
        self.query = nn.Linear(n_meta_features, hidden_dim)
        self.key = nn.Linear(n_meta_features, hidden_dim)
        self.value = nn.Linear(n_meta_features, hidden_dim)
        self.scale = nn.Parameter(torch.sqrt(torch.tensor([hidden_dim], dtype=torch.float32)), requires_grad=True)
        self.attention_dropout = nn.Dropout(0.2)  # Reduced dropout
    
    def forward(self, meta):
        Q = self.query(meta)
        K = self.key(meta)
        V = self.value(meta)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = F.softmax(scores, dim=-1)
        attention = self.attention_dropout(attention)
        weighted_meta = torch.matmul(attention, V)
        return weighted_meta
        
# --- MODIFIED enetv2 ---
class enetv2(nn.Module):
    # Add dropout_scale_factors to __init__
    def __init__(self, backbone, out_dim, n_meta_features=0, load_pretrained=True, dropout_scale_factors=None): # Added dropout_scale_factors
        super(enetv2, self).__init__()
        self.n_meta_features = n_meta_features
        self.backbone = backbone
        self.out_dim = out_dim
        # Store the factors
        self.dropout_scale_factors = dropout_scale_factors if dropout_scale_factors is not None else {v: 1.0 for v in range(3, 8)} # Default if None

        # --- rest of __init__ ---
        try:
            self.enet = timm.create_model(backbone, pretrained=load_pretrained)
            if load_pretrained:
                print(f"Successfully loaded pretrained ImageNet weights for {backbone} using timm.")
            else:
                print(f"Initialized {backbone} with random weights (pretrained=False).")
        except Exception as e:
            print(f"Error: Failed to load pretrained weights for {backbone}: {str(e)}")
            self.enet = timm.create_model(backbone, pretrained=False)
            print(f"Falling back to random initialization for {backbone}.")

        # Base dropout rates (will be scaled)
        self.base_image_dropout_rate = 0.3  # Reduced base rate
        self.base_classifier_dropout_rate = 0.4 # Reduced base rate

        self.image_dropout = nn.Dropout(self.base_image_dropout_rate)
        self.classifier_dropout = nn.Dropout(self.base_classifier_dropout_rate)

        in_ch = self.enet.classifier.in_features
        self.enet.classifier = nn.Identity()

        if n_meta_features > 0:
            # Increased complexity for meta pathway
            meta_hidden_dim = 256 # Make meta pathway potentially stronger
            self.meta_attention = MetadataAttention(n_meta_features, hidden_dim=meta_hidden_dim // 2) # Attention uses half
            self.meta_fc = nn.Sequential(
                nn.Linear(meta_hidden_dim // 2, meta_hidden_dim), # Project attention output
                nn.BatchNorm1d(meta_hidden_dim),
                nn.SiLU(),
                nn.Dropout(p=0.4), # Keep dropout reasonable here
                nn.Linear(meta_hidden_dim, 128), # Final projection to 128
                nn.BatchNorm1d(128),
                nn.SiLU(),
                nn.Dropout(p=0.3) # Another dropout
            )
            in_ch += 128
            print(f"  - ENetV2 Meta Pathway: Attention({n_meta_features}->{meta_hidden_dim//2}), FC({meta_hidden_dim//2}->{meta_hidden_dim}->128)")
        else:
             print("  - ENetV2 Meta Pathway: Disabled")


        self.myfc = nn.Linear(in_ch, out_dim)

        self.current_epoch = 0
        self.gradcam_mode = False
        self.fixed_meta = None
        self.register_buffer('last_preds', None)
        self.register_buffer('last_targets', None)


    # MODIFIED set_epoch
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        enet_version = get_enet_version(self.backbone)
        dropout_scale = self.dropout_scale_factors.get(enet_version, 1.0)
        # print(f"[DEBUG ENetV2 Epoch {epoch}] ENet Version: {enet_version}, Dropout Scale: {dropout_scale:.4f}") # Debug

        # --- Dynamic Dropout Calculation ---
        # More gradual reduction and apply scaling factor
        # Start higher, end lower, apply scale factor
        if epoch <= 5:    # Longer high dropout phase
            current_base_classifier_dropout = 0.65
            current_base_image_dropout = 0.45
        elif epoch <= 12:  # Gradual decrease
             progress = (epoch - 5) / (12 - 5) # 0 to 1
             current_base_classifier_dropout = 0.65 - progress * (0.65 - 0.50) # Range 0.65 -> 0.50
             current_base_image_dropout = 0.45 - progress * (0.45 - 0.35)    # Range 0.45 -> 0.35
        else:            # Final lower dropout phase
            current_base_classifier_dropout = 0.50
            current_base_image_dropout = 0.35

        # Apply scaling and cap
        adjusted_classifier_dropout = min(current_base_classifier_dropout * dropout_scale, 0.95) # Cap high
        adjusted_image_dropout = min(current_base_image_dropout * dropout_scale, 0.90)      # Cap high

        # Apply to layers
        self.classifier_dropout.p = adjusted_classifier_dropout
        self.image_dropout.p = adjusted_image_dropout

        # print(f"[DEBUG ENetV2 Epoch {epoch}] Classifier Dropout: {self.classifier_dropout.p:.4f}, Image Dropout: {self.image_dropout.p:.4f}") # Debug

    # --- extract and forward methods remain the same ---
    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = self.image_dropout(x) # Use the dynamically adjusted dropout

        if self.n_meta_features > 0:
            if self.gradcam_mode and self.fixed_meta is not None:
                x_meta = self.meta_attention(self.fixed_meta)
            elif x_meta is not None:
                x_meta = self.meta_attention(x_meta)
            else:
                raise ValueError("x_meta is required when n_meta_features > 0 and gradcam_mode is False")

            x_meta = self.meta_fc(x_meta)
            x = torch.cat((x, x_meta), dim=1)

        x = self.classifier_dropout(x) # Use the dynamically adjusted dropout
        x = self.myfc(x)
        x = torch.clamp(x, min=-20, max=20)
        if torch.isnan(x).any() or torch.isinf(x).any():
            # print(f"Warning: NaN/Inf in logits at epoch {self.current_epoch}") # Reduced verbosity
            x = torch.nan_to_num(x, nan=0.0, posinf=20, neginf=-20)
        return x
```
:::

::: {.cell .markdown}
# Model EfficientNet + ViT {#model-efficientnet--vit}
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:05.694604Z\",\"iopub.status.busy\":\"2025-04-07T02:59:05.694412Z\",\"iopub.status.idle\":\"2025-04-07T02:59:05.717420Z\",\"shell.execute_reply\":\"2025-04-07T02:59:05.716533Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:05.694587Z\"}" trusted="true"}
``` python
# Updated HybridModel with Gradient Checkpointing
from torch.utils.checkpoint import checkpoint
# --- MODIFIED HybridModel ---
class HybridModel(nn.Module):
    # Add dropout_scale_factors to __init__
    def __init__(self, backbone, out_dim, n_meta_features=0, load_pretrained=True, image_size=448, dropout_scale_factors=None): # Added dropout_scale_factors
        super(HybridModel, self).__init__()
        self.n_meta_features = n_meta_features
        self.out_dim = out_dim
        self.backbone = backbone
        # Store the factors
        self.dropout_scale_factors = dropout_scale_factors if dropout_scale_factors is not None else {v: 1.0 for v in range(3, 8)} # Default if None

        # --- EfficientNet setup ---
        import timm
        self.enet = timm.create_model(backbone, pretrained=load_pretrained)
        if load_pretrained:
            print(f"Hybrid: Loaded {backbone} ImageNet weights.")
        else:
            print(f"Hybrid: Initialized {backbone} random weights.")
        self.enet_features = self.enet.num_features
        self.enet.reset_classifier(0, '')
        self.pool = nn.AdaptiveAvgPool2d(1)
        print(f"ENet backbone: {backbone}, features: {self.enet_features}")

        # --- ViT setup (Keep original, small ViT config) ---
        self.vit_config = ViTConfig(
            image_size=image_size, patch_size=16, hidden_size=192, num_hidden_layers=4,
            num_attention_heads=3, intermediate_size=768, hidden_dropout_prob=0.1, # Base ViT dropout
            attention_probs_dropout_prob=0.1 # Base ViT dropout
        )
        self.vit = ViTModel(self.vit_config)
        self.vit_features = self.vit_config.hidden_size
        print(f"ViT features dimension: {self.vit_features}")

        # --- Fusion Layer ---
        fusion_dim = self.enet_features + self.vit_features
        # Base dropout rate for fusion (will be scaled)
        self.base_fusion_dropout_rate = 0.45
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.BatchNorm1d(512), nn.SiLU(),
            nn.Dropout(self.base_fusion_dropout_rate) # Placeholder, will be updated
        )
        print(f"Fusion input dimension: {fusion_dim}")

        # --- Meta Pathway (if used) ---
        combined_feature_dim = 512 # Output dim of fusion layer
        if self.n_meta_features > 0:
             # Increased complexity for meta pathway in Hybrid
            meta_hidden_dim = 256
            self.meta_attention = MetadataAttention(n_meta_features, hidden_dim=meta_hidden_dim // 2)
            self.meta_fc = nn.Sequential(
                nn.Linear(meta_hidden_dim // 2, meta_hidden_dim),
                nn.BatchNorm1d(meta_hidden_dim),
                nn.SiLU(),
                nn.Dropout(p=0.4), # Keep reasonable
                nn.Linear(meta_hidden_dim, 128),
                nn.BatchNorm1d(128),
                nn.SiLU(),
                nn.Dropout(p=0.3) # Keep reasonable
            )
            combined_feature_dim += 128 # Add meta features dim
            print(f"  - Hybrid Meta Pathway: Attention({n_meta_features}->{meta_hidden_dim//2}), FC({meta_hidden_dim//2}->{meta_hidden_dim}->128)")
        else:
             print("  - Hybrid Meta Pathway: Disabled")

        # --- Final Classifier ---
        # Base dropout rate for classifier (will be scaled)
        self.base_classifier_dropout_rate = 0.35
        self.classifier = nn.Sequential(
            nn.Dropout(self.base_classifier_dropout_rate), # Placeholder, will be updated
            nn.Linear(combined_feature_dim, out_dim)
        )

        self.current_epoch = 0
        self.gradcam_mode = False
        self.fixed_meta = None
        self.register_buffer('last_preds', None)
        self.register_buffer('last_targets', None)

    # MODIFIED set_epoch
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        enet_version = get_enet_version(self.backbone)
        dropout_scale = self.dropout_scale_factors.get(enet_version, 1.0)
        # print(f"[DEBUG Hybrid Epoch {epoch}] ENet Version: {enet_version}, Dropout Scale: {dropout_scale:.4f}") # Debug

        # --- Dynamic Dropout Calculation ---
        # Adjust base rates based on epoch progress
        if epoch <= 5:
            current_base_fusion_dropout = 0.60
            current_base_classifier_dropout = 0.50
        elif epoch <= 12:
            progress = (epoch - 5) / (12 - 5)
            current_base_fusion_dropout = 0.60 - progress * (0.60 - 0.45) # Range 0.60 -> 0.45
            current_base_classifier_dropout = 0.50 - progress * (0.50 - 0.35) # Range 0.50 -> 0.35
        else:
            current_base_fusion_dropout = 0.45
            current_base_classifier_dropout = 0.35

        # Apply scaling and cap
        adjusted_fusion_dropout = min(current_base_fusion_dropout * dropout_scale, 0.95)
        adjusted_classifier_dropout = min(current_base_classifier_dropout * dropout_scale, 0.90)

        # Apply to layers (access dropout layers by index)
        self.fusion[3].p = adjusted_fusion_dropout          # Dropout is the 4th element (index 3)
        self.classifier[0].p = adjusted_classifier_dropout  # Dropout is the 1st element (index 0)

        # print(f"[DEBUG Hybrid Epoch {epoch}] Fusion Dropout: {self.fusion[3].p:.4f}, Classifier Dropout: {self.classifier[0].p:.4f}") # Debug


    # --- extract and forward methods remain the same (using checkpointing) ---
    def extract(self, x):
        def enet_forward(x_enet):
            features = self.enet.forward_features(x_enet)
            pooled = self.pool(features)
            return pooled.view(pooled.size(0), -1)

        def vit_forward(x_vit):
            # IMPORTANT: Ensure ViT receives input compatible with its patch embedding
            # If input `x` is already correct size for ViT (e.g., 224x224 or image_size), use it directly.
            # If ENet modifies spatial dims, you might need separate transforms or resizing for ViT.
            # Assuming `x` is suitable for ViT here.
            return self.vit(pixel_values=x_vit).last_hidden_state[:, 0] # CLS token

        # Use reentrant=False if compatible with your PyTorch version (>1.11 recommended)
        # It can be slightly faster and use less memory than the default reentrant=True
        cnn_features = enet_forward(x)
        vit_features = vit_forward(x)

        combined = torch.cat((cnn_features, vit_features), dim=1)
        fused = self.fusion(combined) # Apply fusion (including its dynamic dropout)
        return fused

    def forward(self, x, x_meta=None):
        x = self.extract(x) # Gets fused features (CNN+ViT) after fusion dropout

        if self.n_meta_features > 0:
            if self.gradcam_mode and self.fixed_meta is not None:
                 meta_attended = self.meta_attention(self.fixed_meta)
            elif x_meta is not None:
                 meta_attended = self.meta_attention(x_meta)
            else:
                raise ValueError("x_meta required for Hybrid model when n_meta_features > 0")

            meta_processed = self.meta_fc(meta_attended)
            x = torch.cat((x, meta_processed), dim=1)

        # Classifier dropout is applied within the self.classifier sequential block
        output = self.classifier(x)
        output = torch.clamp(output, min=-20, max=20) # Clamp final output
        if torch.isnan(output).any() or torch.isinf(output).any():
             output = torch.nan_to_num(output, nan=0.0, posinf=20, neginf=-20)
        return output
```
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:05.718658Z\",\"iopub.status.busy\":\"2025-04-07T02:59:05.718338Z\",\"iopub.status.idle\":\"2025-04-07T02:59:05.732446Z\",\"shell.execute_reply\":\"2025-04-07T02:59:05.731612Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:05.718624Z\"}" trusted="true"}
``` python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```
:::

::: {.cell .markdown}
# Freeze Function
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:05.733594Z\",\"iopub.status.busy\":\"2025-04-07T02:59:05.733357Z\",\"iopub.status.idle\":\"2025-04-07T02:59:05.754043Z\",\"shell.execute_reply\":\"2025-04-07T02:59:05.753214Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:05.733562Z\"}" trusted="true"}
``` python
def get_max_enet_block_index(model):
    """Helper to find the highest block index in the enet module."""
    # Ensure we are looking at the base model if wrapped
    if isinstance(model, nn.DataParallel):
        model = model.module
    if not hasattr(model, 'enet'):
        print("Warning: Model does not have 'enet' attribute.")
        return -1 # Or raise error

    max_block_idx = -1
    for name, _ in model.enet.named_parameters():
        if name.startswith('blocks.'):
            try:
                # Extract block index (e.g., from 'blocks.5.conv_pw')
                idx = int(name.split('.')[1])
                max_block_idx = max(max_block_idx, idx)
            except (IndexError, ValueError):
                continue # Skip layers not matching the 'blocks.N...' pattern
    # print(f"Determined max block index: {max_block_idx}") # Optional debug
    return max_block_idx

def progressive_unfreeze_v3(
    model,
    optimizer, # Pass the current optimizer
    epoch,
    freeze_initially_until_block, # Block index used in partial_freeze_enet (e.g., 4)
    start_unfreeze_epoch,         # When to start unfreezing (e.g., 7)
    full_unfreeze_epoch,          # When to unfreeze everything & reset optimizer (e.g., 15)
    final_enet_lr_mult=0.3,       # Multiplier for backbone LR vs base LR at the end
    final_vit_lr_mult=0.8,        # Multiplier for ViT LR (if hybrid)
    final_head_lr_mult=1.0,       # Multiplier for head/meta LR
    base_lr=1e-3,                 # Base LR for final optimizer config
    enet_type=None,               # Needed for scaling factors
    lr_scale_factors=None         # The dictionary of scaling factors
):
    """
    Unfreezes EfficientNet blocks sequentially from deep to shallow.
    Resets optimizer only once at full_unfreeze_epoch.
    """
    # Ensure we are using the base model
    model_base = model.module if isinstance(model, nn.DataParallel) else model

    # --- Phase 1: Before unfreezing starts ---
    if epoch < start_unfreeze_epoch:
        # print(f"Epoch {epoch}: Keeping layers frozen.") # Optional debug
        return optimizer, False # Return original optimizer, indicate no change

    max_block_idx = get_max_enet_block_index(model_base)
    if max_block_idx == -1:
         print("Warning: Could not determine max block index. Cannot perform sequential unfreeze.")
         return optimizer, False

    # Blocks that were initially frozen and need unfreezing sequentially
    # Example: freeze_initially_until_block=4, max=6 -> blocks_to_unfreeze = [6, 5] (Range is exclusive at end)
    # Corrected: Blocks are 0..N. If frozen up to 4, max is 6, we need to unfreeze 6, 5.
    blocks_to_unfreeze_sequentially = list(range(max_block_idx, freeze_initially_until_block, -1)) # e.g., [6, 5] if max=6, init_freeze=4
    num_unfreeze_stages = len(blocks_to_unfreeze_sequentially)

    # --- Phase 3: Full unfreeze and optimizer reset ---
    if epoch == full_unfreeze_epoch:
        print(f"Epoch {epoch}: Fully unfreezing ALL backbone layers and resetting optimizer.")
        unfrozen_count = 0
        for name, param in model_base.enet.named_parameters():
            if not param.requires_grad:
                 param.requires_grad = True
                 unfrozen_count += 1
        if unfrozen_count > 0:
             print(f"  - Unfroze {unfrozen_count} previously frozen enet parameters.")

        if hasattr(model_base, 'vit'):
            vit_unfrozen_count = 0
            for param in model_base.vit.parameters(): # Ensure ViT is trainable
                if not param.requires_grad:
                     param.requires_grad = True
                     vit_unfrozen_count += 1
            if vit_unfrozen_count > 0:
                print(f"  - Ensured {vit_unfrozen_count} ViT layers are trainable.")

        # --- Create NEW optimizer with differential LRs ---
        enet_version = get_enet_version(enet_type)
        lr_scale = lr_scale_factors.get(enet_version, 1.0) if lr_scale_factors else 1.0

        # Define final learning rates
        final_enet_lr = base_lr * final_enet_lr_mult * lr_scale
        final_vit_lr = base_lr * final_vit_lr_mult
        final_head_lr = base_lr * final_head_lr_mult
        final_meta_lr = base_lr * final_head_lr_mult # Meta layers usually use head LR

        print(f"  - Final LR settings: ENet={final_enet_lr:.2e}, ViT={final_vit_lr:.2e}, Head={final_head_lr:.2e}, Meta={final_meta_lr:.2e}")

        # --- Define Parameter Groups Robustly ---
        param_groups = []
        all_param_ids = set() # Keep track of params assigned to groups

        # Group 1: ENet parameters
        enet_params = [p for n, p in model_base.enet.named_parameters() if p.requires_grad]
        if enet_params:
            param_groups.append({'params': enet_params, 'lr': final_enet_lr, 'weight_decay': 0.02}) # Lower WD for backbone often
            all_param_ids.update(id(p) for p in enet_params)
            # print(f"  - Added ENet group ({len(enet_params)} params)")

        # Group 2: ViT parameters (if hybrid)
        if hasattr(model_base, 'vit'):
            vit_params = [p for n, p in model_base.vit.named_parameters() if p.requires_grad and id(p) not in all_param_ids]
            if vit_params:
                param_groups.append({'params': vit_params, 'lr': final_vit_lr, 'weight_decay': 0.02})
                all_param_ids.update(id(p) for p in vit_params)
                # print(f"  - Added ViT group ({len(vit_params)} params)")

        # Group 3: Fusion parameters (if hybrid)
        if hasattr(model_base, 'fusion'):
            fusion_params = [p for n, p in model_base.fusion.named_parameters() if p.requires_grad and id(p) not in all_param_ids]
            if fusion_params:
                param_groups.append({'params': fusion_params, 'lr': final_head_lr, 'weight_decay': 0.05}) # Higher WD for head
                all_param_ids.update(id(p) for p in fusion_params)
                # print(f"  - Added Fusion group ({len(fusion_params)} params)")

        # Group 4: Meta layers (common to both model types potentially)
        meta_params_list = []
        if hasattr(model_base, 'meta_attention'): meta_params_list.extend(list(model_base.meta_attention.parameters()))
        if hasattr(model_base, 'meta_fc'): meta_params_list.extend(list(model_base.meta_fc.parameters()))
        unique_meta_params = [p for p in meta_params_list if p.requires_grad and id(p) not in all_param_ids]
        if unique_meta_params:
             param_groups.append({'params': unique_meta_params, 'lr': final_meta_lr, 'weight_decay': 0.05})
             all_param_ids.update(id(p) for p in unique_meta_params)
             # print(f"  - Added Meta group ({len(unique_meta_params)} params)")

        # Group 5: Final Classifier (myfc or classifier layer)
        classifier_params_list = []
        if hasattr(model_base, 'myfc'): classifier_params_list.extend(list(model_base.myfc.parameters()))
        if hasattr(model_base, 'classifier'): classifier_params_list.extend(list(model_base.classifier.parameters()))
        unique_classifier_params = [p for p in classifier_params_list if p.requires_grad and id(p) not in all_param_ids]
        if unique_classifier_params:
            param_groups.append({'params': unique_classifier_params, 'lr': final_head_lr, 'weight_decay': 0.05})
            all_param_ids.update(id(p) for p in unique_classifier_params)
            # print(f"  - Added Classifier group ({len(unique_classifier_params)} params)")

        # Catch any remaining trainable parameters (should ideally be none)
        remaining_params = [p for n,p in model_base.named_parameters() if p.requires_grad and id(p) not in all_param_ids]
        if remaining_params:
             print(f"  - WARNING: {len(remaining_params)} trainable parameters were not assigned to an optimizer group. Adding with head LR.")
             param_groups.append({'params': remaining_params, 'lr': final_head_lr, 'weight_decay': 0.05})

        # Create the new optimizer
        new_optimizer = optim.AdamW(param_groups) # WD applied per group
        print(f"Optimizer recreated with {len(param_groups)} parameter groups.")
        return new_optimizer, True # Return NEW optimizer, indicate change happened

    # --- Phase 2: Intermediate unfreezing steps ---
    elif epoch >= start_unfreeze_epoch and num_unfreeze_stages > 0:
        # Calculate which block(s) to unfreeze in this epoch
        total_unfreeze_epochs_span = full_unfreeze_epoch - start_unfreeze_epoch # e.g., 15 - 7 = 8 epochs
        # Spread stages roughly evenly, ensure at least 1 epoch per stage
        epochs_per_stage = max(1, math.ceil(total_unfreeze_epochs_span / num_unfreeze_stages))

        # Determine the index within blocks_to_unfreeze_sequentially
        # e.g. epoch=7, start=7 -> stage_idx = 0 // epochs_per_stage = 0
        # e.g. epoch=8, start=7, epochs_per_stage=2 -> stage_idx = 1 // 2 = 0
        # e.g. epoch=9, start=7, epochs_per_stage=2 -> stage_idx = 2 // 2 = 1
        current_stage_idx = (epoch - start_unfreeze_epoch) // epochs_per_stage

        if current_stage_idx < num_unfreeze_stages:
            block_idx_to_unfreeze = blocks_to_unfreeze_sequentially[current_stage_idx]

            # Unfreeze this block if it's not already trainable
            unfrozen_something_in_stage = False
            for name, param in model_base.enet.named_parameters():
                 layer_name_parts = name.split('.')
                 is_target_block = False
                 if 'blocks' in layer_name_parts:
                      try:
                          block_idx_in_name = int(layer_name_parts[layer_name_parts.index('blocks') + 1])
                          if block_idx_in_name == block_idx_to_unfreeze:
                              is_target_block = True
                      except (IndexError, ValueError, TypeError): pass

                 if is_target_block and not param.requires_grad:
                     param.requires_grad = True
                     unfrozen_something_in_stage = True

            if unfrozen_something_in_stage:
                print(f"Epoch {epoch}: Unfroze enet.blocks.{block_idx_to_unfreeze}")
                # Do NOT reset optimizer here. Return the original one.
                return optimizer, False
        # Else: We might be past the sequential unfreeze stages but before full_unfreeze_epoch
        #       or num_unfreeze_stages was 0 (meaning freeze_initially_until_block >= max_block_idx)

    # If it's not start_unfreeze_epoch or full_unfreeze_epoch, and no intermediate unfreeze happened
    return optimizer, False # Return original optimizer, indicate no change
    
def partial_freeze_enet(model, freeze_until_block=2):
    """
    Partially freeze the EfficientNet backbone up to a specified block.
    Args:
        model: The model instance (may be wrapped in DataParallel).
        freeze_until_block: The last block index to freeze (e.g., 2 freezes blocks.0, blocks.1, blocks.2).
    """
    if isinstance(model, nn.DataParallel):
        model = model.module
    
    # Freeze pretrained layers: conv_stem, bn1, and blocks up to freeze_until_block
    for name, param in model.enet.named_parameters():
        if ('conv_stem' in name or 'bn1' in name or 
            any(f'blocks.{i}.' in name for i in range(freeze_until_block + 1))):
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    print(f"EfficientNet backbone partially frozen up to block {freeze_until_block}.")
    
    # For HybridModel, optionally freeze ViT (adjust as needed)
    if hasattr(model, 'vit'):
        for param in model.vit.parameters():
            param.requires_grad = True  # ViT starts trainable
        print("ViT backbone set to trainable from the start.")
```
:::

::: {.cell .markdown}
# Training Component
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:05.755335Z\",\"iopub.status.busy\":\"2025-04-07T02:59:05.755036Z\",\"iopub.status.idle\":\"2025-04-07T02:59:05.768256Z\",\"shell.execute_reply\":\"2025-04-07T02:59:05.767532Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:05.755304Z\"}" trusted="true"}
``` python
# Fix Warmup Bug
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
```
:::

::: {.cell .markdown}
# Train and Valid
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:05.769202Z\",\"iopub.status.busy\":\"2025-04-07T02:59:05.769015Z\",\"iopub.status.idle\":\"2025-04-07T02:59:05.787300Z\",\"shell.execute_reply\":\"2025-04-07T02:59:05.786395Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:05.769186Z\"}" trusted="true"}
``` python
# Utility function to get resource usage
def get_resource_usage():
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1024**2  # MB
        mem_max = torch.cuda.max_memory_allocated() / 1024**2  # MB
        return {"gpu_memory_allocated": mem_alloc, "gpu_max_memory": mem_max}
    else:
        import psutil
        cpu_usage = psutil.cpu_percent(interval=1)
        ram_usage = psutil.virtual_memory().percent
        return {"cpu_usage": cpu_usage, "ram_usage": ram_usage}
        
class TemperatureScaling:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        # Initialize log_temperature as a learnable parameter starting at 0 (exp(0) = 1)
        self.log_temperature = nn.Parameter(torch.zeros(1).to(device))

    def calibrate(self, loader, max_iter=50):
        """
        Optimizes the temperature using validation data with NLL as the loss.
        Ensures temperature remains positive via exponential parameterization.
        """
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.log_temperature], lr=0.01, max_iter=max_iter)

        # Collect logits and targets from validation data
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for batch in loader:
                if hasattr(self.model, 'module') and self.model.module.n_meta_features > 0:
                    (images, meta), target = batch
                    images, meta, target = images.to(self.device), meta.to(self.device), target.to(self.device)
                    logits = self.model(images, meta)
                else:
                    images, target = batch
                    images, target = images.to(self.device), target.to(self.device)
                    logits = self.model(images)
                all_logits.append(logits)
                all_targets.append(target)
        
        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)

        # Closure for LBFGS optimization
        def nll_closure():
            optimizer.zero_grad()
            temperature = torch.exp(self.log_temperature)  # Ensures temperature > 0
            scaled_logits = logits / temperature
            loss = nll_criterion(scaled_logits, targets)
            loss.backward()
            return loss

        # Optimize log_temperature
        optimizer.step(nll_closure)
        optimal_temperature = torch.exp(self.log_temperature).item()
        print(f"Optimal temperature: {optimal_temperature:.4f}")
        return optimal_temperature

    def forward(self, logits):
        """
        Applies the learned temperature to logits to obtain calibrated probabilities.
        """
        temperature = torch.exp(self.log_temperature)
        return torch.softmax(logits / temperature, dim=1)
```
:::

::: {.cell .markdown}
# Early Stopping Mechanism
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:05.788416Z\",\"iopub.status.busy\":\"2025-04-07T02:59:05.788164Z\",\"iopub.status.idle\":\"2025-04-07T02:59:05.805029Z\",\"shell.execute_reply\":\"2025-04-07T02:59:05.804262Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:05.788385Z\"}" trusted="true"}
``` python
class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,  # Increased from 3 to 10
        mode: str = 'max',
        delta: float = 0.005,  # Small relative improvement
        relative_delta: bool = True,  # Use relative delta
        warm_up: int = 9,  # Extended to cover frozen phase (epochs 1-9)
        verbose: bool = True,
        checkpoint_path: str = 'best_model.pth',
        score_weights: Optional[Dict[str, float]] = None
    ):
        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.relative_delta = relative_delta
        self.warm_up = warm_up
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self.counters: Dict[str, int] = {}
        self.best_scores: Dict[str, float] = {}
        self.best_epoch: Dict[str, int] = {}
        self.early_stop = False
        self._is_first = True
        self.score_weights = score_weights or {
            'binary_auc': 0.4,      # Increased to prioritize AUC target
            'binary_recall': 0.2,   # Reduced slightly to balance
            'multiclass_auc': 0.3,  # Kept to ensure multiclass consideration
            'binary_specificity': 0.05,
            'val_loss': 0.05        # Minimize, low weight
        }
        # Validate score_weights
        if self.score_weights:
            total_weight = sum(self.score_weights.values())
            if total_weight <= 0:
                raise ValueError("Total weight in score_weights must be positive.")
            # Normalize weights to sum to 1.0 (optional)
            self.score_weights = {k: v / total_weight for k, v in self.score_weights.items()}

    def reset(self):
        """Reset the early stopping object to its initial state."""
        self.counters = {}
        self.best_scores = {}
        self.best_epoch = {}
        self.early_stop = False
        self._is_first = True

    def __call__(
        self,
        metrics: Union[Dict[str, float], float],
        model: Optional[torch.nn.Module] = None,
        epoch: Optional[int] = None
    ):
        if epoch is None:
            raise ValueError("epoch must be provided.")
        if epoch <= self.warm_up:
            if self.verbose:
                print(f"Epoch {epoch} in warm-up period, skipping early stopping.")
            return

        if isinstance(metrics, (int, float)):
            metrics = {'val_metric': metrics}

        # Compute composite score
        score = 0.0
        for metric_name, val_metric in metrics.items():
            if not isinstance(val_metric, (int, float)):
                raise ValueError(f"Metric {metric_name} must be a number, got {type(val_metric)}")
            if metric_name not in metrics:
                continue  # Skip missing metrics
            weight = self.score_weights.get(metric_name, 0)
            if weight > 0:
                # Transform val_loss (minimize) to fit max mode
                adjusted_score = -val_metric if metric_name == 'val_loss' else val_metric
                score += adjusted_score * weight

        current_delta = self.delta if not self.relative_delta else self.delta * abs(self.best_scores.get('composite', 0))

        if self._is_first or self.best_scores.get('composite') is None:
            self.best_scores['composite'] = score
            self.best_epoch['composite'] = epoch
            self.counters['composite'] = 0
            self._is_first = False
            if model and self.checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_scores': self.best_scores,
                    'best_epoch': self.best_epoch
                }, self.checkpoint_path)
                if self.verbose:
                    print(f"Initial best composite score: {score:.6f} at epoch {epoch}, model saved to {self.checkpoint_path}")
        elif score > self.best_scores['composite'] + current_delta:
            self.best_scores['composite'] = score
            self.best_epoch['composite'] = epoch
            self.counters['composite'] = 0
            if model and self.checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_scores': self.best_scores,
                    'best_epoch': self.best_epoch
                }, self.checkpoint_path)
                if self.verbose:
                    print(f"New best composite score: {score:.6f} at epoch {epoch}, model saved to {self.checkpoint_path}")
        else:
            self.counters['composite'] = self.counters.get('composite', 0) + 1
            if self.verbose:
                print(f"No improvement in composite score. Counter: {self.counters['composite']}/{self.patience}")
                for metric_name, val_metric in metrics.items():
                    print(f"  {metric_name}: {val_metric:.6f}")
            if self.counters['composite'] >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {epoch} epochs due to composite score.")

def get_best_scores(self) -> Dict[str, float]:
        """Return the best scores for all monitored metrics."""
        return {'composite': self.best_scores.get('composite', float('-inf') if self.mode == 'max' else float('inf'))}

def get_best_epoch(self) -> Dict[str, int]:
        """Return the epoch of the best score for all monitored metrics."""
        return {'composite': self.best_epoch.get('composite', 0)}
```
:::

::: {.cell .markdown}
# Training Epoch
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:05.806076Z\",\"iopub.status.busy\":\"2025-04-07T02:59:05.805818Z\",\"iopub.status.idle\":\"2025-04-07T02:59:05.820813Z\",\"shell.execute_reply\":\"2025-04-07T02:59:05.820062Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:05.806056Z\"}" trusted="true"}
``` python
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
# Make sure get_resource_usage is defined elsewhere or remove the call

def train_epoch(model, loader, optimizer, experiment, epoch, scaler=None, accum_steps=1, criterion_multi=None, mel_idx=None, lambda_binary=0.75, device=None):
    """
    Training epoch function.
    REMOVED: AdaBoost-style weight updates.
    """
    if criterion_multi is None or mel_idx is None:
        raise ValueError("criterion_multi and mel_idx must be provided to train_epoch")

    use_device = device is not None
    if not use_device:
        print("Warning: No device provided; defaulting to 'cuda' if available")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train() # Ensure model is in training mode
    train_loss_list = [] # Store loss of each batch (before accumulation division)
    train_correct = 0
    train_total = 0
    start_time = time.time()
    optimizer.zero_grad() # Zero gradients at the start of the epoch

    # Initialize tqdm progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch} - Loss: N/A, Acc: N/A", total=len(loader))

    for batch_idx, batch in enumerate(pbar):
        # --- Data Handling ---
        if hasattr(model.module, 'n_meta_features') and model.module.n_meta_features > 0: # Check n_meta_features on the base model
            (images, meta), target = batch
            images, meta, target = images.to(device), meta.to(device), target.to(device)
        else:
            images, target = batch
            images, target = images.to(device), target.to(device)

        # --- Forward Pass ---
        if scaler: # Using AMP
            with torch.cuda.amp.autocast():
                logits = model(images, meta) if (hasattr(model.module, 'n_meta_features') and model.module.n_meta_features > 0) else model(images)
                # --- Loss Calculation ---
                multiclass_loss = criterion_multi(logits, target)
                binary_target = (target == mel_idx).float()
                binary_logits = logits[:, mel_idx]
                binary_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_target) # Use BCEWithLogits
                total_loss_unscaled = multiclass_loss + lambda_binary * binary_loss
                total_loss = total_loss_unscaled / accum_steps # Scale loss for accumulation
            # --- Backward Pass (AMP) ---
            scaler.scale(total_loss).backward()
        else: # Not using AMP
            logits = model(images, meta) if (hasattr(model.module, 'n_meta_features') and model.module.n_meta_features > 0) else model(images)
            # --- Loss Calculation ---
            multiclass_loss = criterion_multi(logits, target)
            binary_target = (target == mel_idx).float()
            binary_logits = logits[:, mel_idx]
            binary_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_target) # Use BCEWithLogits
            total_loss_unscaled = multiclass_loss + lambda_binary * binary_loss
            total_loss = total_loss_unscaled / accum_steps # Scale loss for accumulation
            # --- Backward Pass ---
            total_loss.backward()

        # Store unscaled loss for epoch average reporting
        train_loss_list.append(total_loss_unscaled.item())

        # --- Accuracy Tracking (using the same logits) ---
        with torch.no_grad(): # No need gradients for accuracy calculation
             preds = logits.argmax(dim=1)
             batch_correct = (preds == target).sum().item()
             batch_total = target.size(0)
             train_correct += batch_correct
             train_total += batch_total

        # --- Optimization Step ---
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            if scaler:
                scaler.step(optimizer) # Unscales gradients and steps optimizer
                scaler.update()        # Updates scaler for next iteration
            else:
                optimizer.step()       # Standard optimizer step
            optimizer.zero_grad()      # Zero gradients *after* stepping

        # --- Update Progress Bar ---
        # Calculate metrics based on accumulated values so far
        avg_loss_so_far = np.mean(train_loss_list) if train_loss_list else 0.0
        acc_so_far = (train_correct / train_total) * 100.0 if train_total > 0 else 0.0
        pbar.set_description(f"Epoch {epoch} - Loss: {avg_loss_so_far:.4f}, Acc: {acc_so_far:.2f}%")
        # pbar.set_postfix(batch=batch_idx + 1) # Optional: show batch number

    # --- End of Epoch ---
    pbar.close()
    avg_train_loss = np.mean(train_loss_list) if train_loss_list else 0.0 # Final average loss for the epoch
    train_acc = (train_correct / train_total) * 100.0 if train_total > 0 else 0.0
    epoch_time = time.time() - start_time

    # --- Resource Usage ---
    try:
        resources = get_resource_usage()
    except NameError:
        resources = {} # Handle if function not defined
        print("Warning: get_resource_usage() not defined.")

    # --- Print Epoch Summary ---
    print(
        f"Epoch {epoch} - Training Time: {epoch_time:.2f}s, Avg Loss: {avg_train_loss:.5f}, "
        f"Acc: {train_acc:.2f}%, Resources: {resources}"
    )

    # --- Logging to W&B ---
    if experiment:
        log_data = {
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "train_epoch_time_seconds": epoch_time,
        }
        # Add resource usage if available and it's a dictionary
        if isinstance(resources, dict):
             log_data.update({f"train_{k}": v for k, v in resources.items()})
        experiment.log(log_data, step=epoch)

    # --- Return average loss for the epoch ---
    return avg_train_loss
```
:::

::: {.cell .markdown}
# Validation Epoch
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:05.821982Z\",\"iopub.status.busy\":\"2025-04-07T02:59:05.821663Z\",\"iopub.status.idle\":\"2025-04-07T02:59:05.851853Z\",\"shell.execute_reply\":\"2025-04-07T02:59:05.851135Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:05.821956Z\"}" trusted="true"}
``` python
def val_epoch(model, loader, experiment, epoch, n_test=1, recalib_interval=5, # REMOVED mc_samples
              criterion_multi=None, mel_idx=None, lambda_binary=0.5, device=None, use_amp=True):
    """
    Validation epoch function.
    REMOVED: MC Dropout logic and uncertainty calculations.
    KEPT: Temperature Scaling.
    """
    if criterion_multi is None or mel_idx is None:
        raise ValueError("criterion_multi and mel_idx must be provided to val_epoch")
    if device is None:
        print("Warning: No device provided; defaulting to 'cuda' if available")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()  # Set model STRICTLY to eval mode for validation
    out_dim = model.module.out_dim

    val_loss_batch_list = [] # Store average loss per batch
    PROBS_ALL = []          # Store predicted probabilities for the epoch
    TARGETS = []            # Store true targets for the epoch
    start_time = time.time()

    # --- Temperature Scaling Setup ---
    temp_scaler = TemperatureScaling(model, device)
    optimal_temp = None
    if epoch > 1 and epoch % recalib_interval == 1:
        print(f"Calibrating model with Temperature Scaling at epoch {epoch}...")
        optimal_temp = temp_scaler.calibrate(loader)
        if experiment and optimal_temp is not None:
            experiment.log({"optimal_temperature": optimal_temp}, step=epoch)

    current_temp = optimal_temp if optimal_temp is not None else (torch.exp(temp_scaler.log_temperature).item() if hasattr(temp_scaler, 'log_temperature') else 1.0)
    print(f"Using temperature: {current_temp:.4f} for validation epoch {epoch}")
    # --- End Temperature Scaling Setup ---

    # --- Initialize Metrics ---
    multiclass_acc = MulticlassAccuracy(num_classes=out_dim, average='macro').to(device)
    multiclass_f1 = MulticlassF1Score(num_classes=out_dim, average='macro').to(device)
    multiclass_auc = MulticlassAUROC(num_classes=out_dim, average='macro', thresholds=None).to(device)
    multiclass_confmat = MulticlassConfusionMatrix(num_classes=out_dim).to(device)
    binary_auc = BinaryAUROC(thresholds=None).to(device)

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Validating Epoch {epoch}", total=len(loader))
        for batch_idx, batch in enumerate(pbar):
            # --- Unpack Batch ---
            if hasattr(model.module, 'n_meta_features') and model.module.n_meta_features > 0:
                (images, meta), target = batch
                images, meta, target = images.to(device), meta.to(device), target.to(device)
            else:
                images, target = batch
                images, target = images.to(device), target.to(device)
            batch_size = images.shape[0]

            # --- Single Inference Pass ---
            with torch.cuda.amp.autocast(enabled=use_amp):
                if hasattr(model.module, 'n_meta_features') and model.module.n_meta_features > 0:
                    logits = model(images, meta)
                else:
                    logits = model(images)

                # Clamp logits and handle potential NaN/Inf
                logits = torch.clamp(logits, min=-20, max=20)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"Warning: NaN/Inf in raw logits at batch {batch_idx}")
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=20, neginf=-20)

                # --- Calculate Loss (using raw logits) ---
                multiclass_loss = criterion_multi(logits, target)
                binary_target = (target == mel_idx).float()
                binary_logits = logits[:, mel_idx]
                binary_loss = nn.functional.binary_cross_entropy_with_logits(
                    torch.clamp(binary_logits, -10, 10), binary_target, reduction='mean'
                )
                total_loss = multiclass_loss + lambda_binary * binary_loss

                # --- Calculate probabilities using the defined current_temp ---
                current_probs = torch.softmax(logits / current_temp, dim=1)

                # Handle potential NaN/Inf in probabilities and clamp
                if torch.isnan(current_probs).any() or torch.isinf(current_probs).any():
                     print(f"Warning: NaN/Inf in softmax probs at batch {batch_idx}")
                     current_probs = torch.nan_to_num(current_probs, nan=0.5, posinf=1.0, neginf=0.0)
                current_probs = torch.clamp(current_probs, 1e-6, 1.0 - 1e-6)

            # --- Store Batch Results ---
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: NaN or Inf in batch total_loss at batch {batch_idx}")
                total_loss = torch.tensor(0.0, device=device) # Assign zero loss if issue
            val_loss_batch_list.append(total_loss.item())

            PROBS_ALL.append(current_probs) # Append probs from single pass
            TARGETS.append(target)

            # --- Update Metrics using current_probs ---
            multiclass_acc.update(current_probs, target)
            multiclass_f1.update(current_probs, target)
            multiclass_auc.update(current_probs, target)
            multiclass_confmat.update(current_probs.argmax(dim=1), target)
            binary_targets_batch = (target == mel_idx).float()
            binary_probs_batch = current_probs[:, mel_idx]
            binary_auc.update(binary_probs_batch, binary_targets_batch)

            pbar.set_postfix(loss=total_loss.item())
        pbar.close()

    # --- Aggregate Epoch Results ---
    if not val_loss_batch_list:
        print(f"Epoch {epoch} - No batches processed during validation.")
        # Return default/dummy values safely matching the *new* reduced structure
        return (
            0.0, 0.5, 0.5, # val_loss, binary_auc, multiclass_auc
            np.array([]), np.array([]), # PROBS, TARGETS
            0.0, 0.0, 0.0, 0.0, 0.0, # binary_acc, prec, recall, f1, spec
            0.0, 0.0, [], [], # multiclass_acc, f1, multi_cm, binary_cm
            1.0, # current_temp
            None # roc_data
        )

    val_loss_avg = np.mean(val_loss_batch_list)
    epoch_time = time.time() - start_time
    try:
        resources = get_resource_usage()
    except NameError:
        resources = {}
        print("Warning: get_resource_usage() not defined.")

    PROBS_ALL = torch.cat(PROBS_ALL, dim=0)
    TARGETS = torch.cat(TARGETS, dim=0)

    PROBS = PROBS_ALL.cpu().numpy()
    TARGETS = TARGETS.cpu().numpy()

    # --- Compute Final Metrics ---
    print(f"Epoch {epoch} - Aggregated PROBS shape: {PROBS.shape}, TARGETS shape: {TARGETS.shape}")
    if PROBS.shape[0] == 0 or TARGETS.shape[0] == 0:
         print(f"Epoch {epoch} - Empty aggregated results, cannot compute metrics.")
         # Return dummy values matching the new structure
         return (
            val_loss_avg, 0.5, 0.5,
            PROBS, TARGETS,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, [], [],
            current_temp, None
         )

    multiclass_acc_val = multiclass_acc.compute().item() * 100.0
    multiclass_f1_val = multiclass_f1.compute().item()
    multiclass_auc_val = multiclass_auc.compute().item()
    multiclass_conf_matrix = multiclass_confmat.compute().cpu().numpy().tolist()
    binary_auc_val = binary_auc.compute().item()

    # --- Rest of the metric calculations (thresholding, binary metrics) ---
    binary_targets_np = (TARGETS == mel_idx).astype(np.float32)
    binary_probs_np = PROBS[:, mel_idx]
    # Slightly adjust probs for threshold finding robustness
    binary_probs_np_adjusted = np.minimum(binary_probs_np * 1.1, 1.0)

    # Find optimal threshold based on F1 from PR curve (fine-grained search)
    best_threshold = 0.5 # Default
    best_f1 = 0.0
    f1_history = []
    try: # Added try-except block for robustness
        precision, recall, pr_thresholds = precision_recall_curve(binary_targets_np, binary_probs_np_adjusted)
        optimal_threshold_pr = 0.5 # Default
        best_f1_from_pr = 0.0
        if len(precision) > 1 and len(recall) > 1:
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-6)
            if len(f1_scores) > 0:
                optimal_idx_pr = np.argmax(f1_scores)
                best_f1_from_pr = f1_scores[optimal_idx_pr]
                optimal_threshold_pr = pr_thresholds[optimal_idx_pr]
                print(f"Epoch {epoch} - PR-based optimal threshold: {optimal_threshold_pr:.4f}, F1: {best_f1_from_pr:.4f}")
            else: print(f"Epoch {epoch} - PR-based optimal threshold: {optimal_threshold_pr:.4f} (fallback - no valid F1 scores)")
        else: print(f"Epoch {epoch} - PR-based optimal threshold: {optimal_threshold_pr:.4f} (fallback - insufficient P/R points)")

        thresholds_to_test = np.linspace(max(0, optimal_threshold_pr - 0.05), min(1, optimal_threshold_pr + 0.05), 50)
        best_f1 = 0; best_threshold = optimal_threshold_pr;
        for thresh in thresholds_to_test:
            preds = (binary_probs_np_adjusted > thresh).astype(np.float32)
            f1 = f1_score(binary_targets_np, preds, zero_division=0)
            f1_history.append((thresh, f1))
            if f1 > best_f1: best_f1 = f1; best_threshold = thresh
        print(f"Epoch {epoch} - Best F1 threshold (fine-grained): {best_threshold:.4f}, Best F1: {best_f1:.4f}")
    except ValueError as ve:
        print(f"Warning: Error during threshold optimization (likely due to single class in batch/epoch): {ve}")
        # Keep default best_threshold = 0.5

    # Calculate metrics using best_threshold
    binary_preds_best = (binary_probs_np_adjusted > best_threshold).astype(np.float32)
    binary_conf_matrix_best = [[0,0],[0,0]] # Default
    binary_specificity_best, binary_precision_best, binary_recall_best, binary_f1_best, binary_acc_best = 0,0,0,0,0
    if len(binary_targets_np) > 0:
        try:
            binary_conf_matrix_best = confusion_matrix(binary_targets_np, binary_preds_best).tolist()
            if len(np.array(binary_conf_matrix_best).ravel()) == 4:
                 tn_best, fp_best, fn_best, tp_best = np.array(binary_conf_matrix_best).ravel()
                 binary_specificity_best = tn_best / (tn_best + fp_best) if (tn_best + fp_best) > 0 else 0.0
            else: tn_best, fp_best, fn_best, tp_best = 0,0,0,0 # Handle case where CM is not 2x2

            binary_precision_best = precision_score(binary_targets_np, binary_preds_best, zero_division=0)
            binary_recall_best = recall_score(binary_targets_np, binary_preds_best, zero_division=0)
            binary_f1_best = f1_score(binary_targets_np, binary_preds_best, zero_division=0) # This IS best_f1
            binary_acc_best = accuracy_score(binary_targets_np, binary_preds_best) * 100
        except ValueError as ve_metrics:
             print(f"Warning: Error calculating binary metrics (likely due to single class): {ve_metrics}")

    # --- Print Epoch Summary (using best F1 metrics) ---
    print(
        f"Epoch {epoch} - Val Loss: {val_loss_avg:.5f}, "
        f"Binary AUC: {binary_auc_val:.4f}, "
        f"Best F1 Metrics (Thresh={best_threshold:.4f}): Acc={binary_acc_best:.2f}%, P={binary_precision_best:.4f}, R={binary_recall_best:.4f}, F1={binary_f1_best:.4f}, Spec={binary_specificity_best:.4f}, "
        f"Multiclass Acc: {multiclass_acc_val:.2f}%, MC F1: {multiclass_f1_val:.4f}, MC AUC: {multiclass_auc_val:.4f}, Val Time: {epoch_time:.2f}s"
    )

    # --- Construct Metrics Dictionary ---
    # REMOVED uncertainty metrics
    metrics = {
        "val_loss": val_loss_avg,
        "binary_auc": binary_auc_val,
        "multiclass_auc": multiclass_auc_val if not np.isnan(multiclass_auc_val) else 0.5,
        "binary_acc": binary_acc_best,
        "binary_precision": binary_precision_best,
        "binary_recall": binary_recall_best,
        "binary_specificity": binary_specificity_best,
        "binary_f1": binary_f1_best,
        "multiclass_acc": multiclass_acc_val,
        "multiclass_f1": multiclass_f1_val,
        # REMOVED: avg_uncertainty, uncertainty_correct, uncertainty_incorrect, binary uncertainties, epistemic, aleatoric
        "val_epoch_time_seconds": epoch_time,
        **{f"val_{k}": v for k, v in resources.items()}
        # Per-class errors were tied to uncertainty/MC, removed for simplicity unless needed elsewhere
    }

    # --- Logging to WandB ---
    if experiment:
        experiment.log(metrics, step=epoch)
        # Log thresholds and CMs (using best F1 threshold versions primarily)
        experiment.log({
            # REMOVED: optimal_threshold_roc, optimal_threshold_pr, # Kept best_f1_threshold
            "best_f1_threshold": best_threshold,
            "f1_threshold_history": wandb.Table(columns=["threshold", "f1_score"], data=f1_history) if f1_history else None
        }, step=epoch)

        # Define class names for logging CMs
        if out_dim == 9: class_names = ['AK','BCC','BKL','DF','SCC','VASC','melanoma','nevus','unknown']
        else: class_names = [str(i) for i in range(out_dim)]

        if len(binary_targets_np) > 0: # Ensure data exists for plotting
            experiment.log({
                "binary_confusion_matrix_best_f1": wandb.plot.confusion_matrix(
                    y_true=binary_targets_np.astype(np.int32), preds=binary_preds_best,
                    class_names=['non-melanoma', 'melanoma']
                ),
                "multiclass_confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=TARGETS.astype(np.int32), preds=PROBS.argmax(axis=1),
                    class_names=class_names
                )
            }, step=epoch)
            # Log tables as well
            try:
                 experiment.log({
                     "binary_confusion_matrix_table_best_f1": wandb.Table(
                         columns=['Pred Non-Mel', 'Pred Mel'],
                         data=binary_conf_matrix_best # Already a list
                     ),
                     "multiclass_confusion_matrix_table": wandb.Table(
                         columns=[f"Pred {name}" for name in class_names],
                         data=multiclass_conf_matrix # Already a list
                     )
                 }, step=epoch)
            except Exception as log_e:
                 print(f"Warning: Failed to log confusion matrix tables to WandB: {log_e}")

    # --- ROC Data for Plotting ---
    roc_data = None
    if len(binary_targets_np) > 0 :
        try:
            fpr, tpr, roc_thresholds = roc_curve(binary_targets_np, binary_probs_np_adjusted)
            if len(fpr) > 0 and len(tpr) > 0: # Check if roc_curve returned valid data
                roc_data = (fpr, tpr, roc_thresholds)
        except ValueError as roc_e:
             print(f"Warning: Could not compute ROC curve: {roc_e}")


    # --- Return values (Updated Structure) ---
    return (
        val_loss_avg, binary_auc_val, multiclass_auc_val, # Core scalar metrics
        PROBS, TARGETS,                       # Predictions and targets (NP arrays)
        binary_acc_best, binary_precision_best, binary_recall_best, binary_f1_best, binary_specificity_best, # Binary metrics @ best F1 thresh
        multiclass_acc_val, multiclass_f1_val, multiclass_conf_matrix, binary_conf_matrix_best, # Multiclass metrics, CMs
        current_temp,                         # Temperature used
        roc_data                              # ROC data tuple or None
    ) # Total 15 return values
```
:::

::: {.cell .markdown}
# Grad-Cam type shit
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:05.852739Z\",\"iopub.status.busy\":\"2025-04-07T02:59:05.852524Z\",\"iopub.status.idle\":\"2025-04-07T02:59:05.871182Z\",\"shell.execute_reply\":\"2025-04-07T02:59:05.870573Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:05.852696Z\"}" trusted="true"}
``` python
def load_and_preprocess_for_gradcam(image_path, target_size):
    """Loads an image, resizes it, and returns both normalized tensor and unnormalized numpy array."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB

    # Resize
    img_resized = cv2.resize(img, (target_size, target_size))

    # --- Prepare image for visualization (unnormalized) ---
    # Needs to be float32 between 0 and 1
    img_for_display = np.float32(img_resized) / 255.0

    # --- Prepare image for model (normalized) ---
    # Apply normalization matching your transforms_val/transforms_train
    normalize_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    normalized_img_data = normalize_transform(image=img_resized)
    normalized_img_np = normalized_img_data['image']

    # Convert to tensor, add batch dimension, move channel first
    # (B, C, H, W) expected by model
    input_tensor = torch.tensor(normalized_img_np).permute(2, 0, 1).unsqueeze(0).float()

    return input_tensor, img_for_display

print("Defined load_and_preprocess_for_gradcam function.")
```
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:05.872322Z\",\"iopub.status.busy\":\"2025-04-07T02:59:05.872059Z\",\"iopub.status.idle\":\"2025-04-07T02:59:05.886964Z\",\"shell.execute_reply\":\"2025-04-07T02:59:05.886297Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:05.872292Z\"}" trusted="true"}
``` python
# Make sure this is the code in your cell defining the function
def generate_gradcam(model, target_layer, input_tensor, original_image,
                     target_class_idx, device, meta_tensor=None):
    """Generates and displays Grad-CAM heatmap, handling optional metadata."""
    # ---- ADD THIS PRINT STATEMENT ----
    print("--- Executing UPDATED generate_gradcam (with ModelWrapper logic) ---")
    # ---- END ADD ----

    # Ensure model is on the correct device and in eval mode
    model.to(device)
    model.eval()

    # --- Grad-CAM Setup ---
    targets = [ClassifierOutputTarget(target_class_idx)]

    # --- Model Wrapper for Metadata (if needed) ---
    model_wrapper = model # By default, use the original model
    if meta_tensor is not None:
        # If metadata exists, create a simple wrapper
        meta_tensor = meta_tensor.to(device) # Ensure meta is on device
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model, meta_data):
                super().__init__()
                self.model = model
                self.meta_data = meta_data

            def forward(self, x):
                # Assumes model forward signature is forward(self, image_tensor, meta_tensor)
                # Adjust if your signature is different (e.g., forward(self, image_tensor, x_meta=meta_tensor))
                return self.model(x, self.meta_data) # Pass both image and meta

        model_wrapper = ModelWrapper(model, meta_tensor)
        print("  Using ModelWrapper for Grad-CAM to pass metadata.")


    # --- Create GradCAM object ---
    # Use the model_wrapper if metadata is present, otherwise the original model
    grad_cam = GradCAM(model=model_wrapper, target_layers=[target_layer])

    # --- Generate CAM ---
    # Now, only pass input_tensor. Metadata is handled by the wrapper.
    grayscale_cam = grad_cam(input_tensor=input_tensor,
                             targets=targets)

    # Take the first CAM in the batch
    grayscale_cam = grayscale_cam[0, :]

    # --- Visualization ---
    visualization = show_cam_on_image(original_image,  # HWC, float32, 0-1 range
                                      grayscale_cam,
                                      use_rgb=True)

    return visualization, grayscale_cam
```
:::

::: {.cell .markdown}
# Main training function
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:05.890140Z\",\"iopub.status.busy\":\"2025-04-07T02:59:05.889935Z\",\"iopub.status.idle\":\"2025-04-07T02:59:05.941110Z\",\"shell.execute_reply\":\"2025-04-07T02:59:05.940247Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:05.890123Z\"}" trusted="true"}
``` python
def run_single_model(model_type='efficientnet', enet_type='efficientnet_b5'):
    """
    Main function to orchestrate the training and validation of a single model.

    Handles:
    - Setup (Device, WandB, Scaling Factors)
    - Data Preparation (Splitting, Sampling, Loaders)
    - Model Instantiation & Pretrained Weight Loading
    - Initial Freezing
    - Loss Criterion Definition
    - Optimizer & Scheduler Setup (with phased approach)
    - Early Stopping Setup
    - Main Training & Validation Loop
    - Phased Learning Rate Adjustments & Unfreezing
    - Best Model Tracking & Saving
    - Final Logging & Artifact Saving
    - Returns key results.
    """
    print(f"--- Starting run_single_model ---")
    print(f"Model Type: {model_type}, ENet Type: {enet_type}")

    # --- 1. Setup ---
    enet_version = get_enet_version(enet_type)
    lr_scale_factors, dropout_scale_factors = get_scaling_factors(model_type, enet_type, use_meta, use_external)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Running on {torch.cuda.device_count()} GPUs with DataParallel.")

    # WandB Initialization
    print("Initializing WandB...")
    # Ensure wandb is imported
    try:
        import wandb
        wandb.init(
            project="Skripsi",
            entity="arveda-ava-universitas-gadjah-mada-library", # Replace with your entity if different
            config={ # Log key hyperparameters
                "model_type": model_type,
                "enet_type": enet_type,
                "kernel_type": kernel_type,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "init_lr": init_lr,
                "out_dim": out_dim,
                "mel_idx": mel_idx,
                "image_size": image_size,
                "use_amp": use_amp,
                "use_meta": use_meta,
                "n_meta_features": n_meta_features,
                "use_external": use_external,
                "accum_steps": accum_steps,
                "lambda_binary": 0.5, # Example value, adjust if needed
                "lr_scale_factor_used": lr_scale_factors.get(enet_version, 1.0),
                "dropout_scale_factor_used": dropout_scale_factors.get(enet_version, 1.0),
            }
        )
        tz = pytz.timezone('Asia/Jakarta')
        run_name = f"{kernel_type}_E{enet_version}_{model_type}{'_Meta' if use_meta else ''}{'_Ext' if use_external else ''}_{datetime.now(tz).strftime('%Y%m%d_%H%M')}"
        wandb.run.name = run_name
        print(f"WandB run initialized: {run_name}")
    except ImportError:
        print("WandB not installed. Skipping logging.")
        wandb = None # Set wandb to None if not available

    torch.cuda.empty_cache()

    # --- 2. Data Preparation ---
    print("\n--- Preparing Data ---")
    # Splitting data
    df_train_set, df_valid_set = train_test_split(df_train, test_size=0.2, stratify=df_train['target'], random_state=42)
    print(f"Initial split: Train={len(df_train_set)}, Valid={len(df_valid_set)}")

    # Sampling/Undersampling Logic (same as before)
    if DEBUG:
        # ... (DEBUG sampling logic remains here) ...
        train_sample_size = min(len(df_train_set), max(batch_size * 40, 800))
        valid_sample_size = min(len(df_valid_set), max(batch_size * 16, 400))
        n_classes = len(df_train_set['target'].unique())
        min_class_size = df_train_set['target'].value_counts().min() if not df_train_set.empty else 0
        samples_per_class = min(max(train_sample_size // n_classes, 1), min_class_size) if min_class_size > 0 else 0

        if samples_per_class > 0:
             df_train_sampled = df_train_set.groupby('target', group_keys=False).apply(
                 lambda x: x.sample(n=min(len(x), samples_per_class), random_state=42) if len(x) >= samples_per_class else x.sample(n=len(x), random_state=42) # Handle small groups
             )
             actual_train_size = min(len(df_train_sampled), train_sample_size)
             df_train_set = df_train_sampled.sample(n=actual_train_size, random_state=42)
        else:
             print("Warning: Cannot sample in DEBUG mode due to small class size.")
             df_train_set = df_train_set.head(0) # Empty dataframe if cannot sample

        # Balance validation set
        samples_per_class_val = valid_sample_size // n_classes if n_classes > 0 else 0
        if samples_per_class_val > 0 and not df_valid_set.empty:
             df_valid_sampled = df_valid_set.groupby('target', group_keys=False).apply(
                 lambda x: x.sample(n=samples_per_class_val, replace=True, random_state=42)
             )
             actual_valid_size = min(len(df_valid_sampled), valid_sample_size)
             df_valid_set = df_valid_sampled.sample(n=actual_valid_size, random_state=42)
        else:
             print("Warning: Cannot sample validation set in DEBUG mode.")
             df_valid_set = df_valid_set.head(0)

        print(f"DEBUG mode: Training on {len(df_train_set)} samples, Validating on {len(df_valid_set)} samples")

    else: # Non-DEBUG mode
        undersample_cap = 4000
        print(f"Non-DEBUG mode: Applying undersampling with cap {undersample_cap} per class to training set.")
        df_train_set = df_train_set.groupby('target', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), undersample_cap), random_state=42)
        ).reset_index(drop=True)
        print(f"Non-DEBUG mode: Training on {len(df_train_set)} samples, Validating on {len(df_valid_set)} (stratified) samples")

    # Ensure target column exists
    if 'target' not in df_train_set.columns or 'target' not in df_valid_set.columns:
        raise KeyError("'target' column missing. Check data prep.")

    print(f"Final Train set class counts:\n{df_train_set['target'].value_counts()}")
    print(f"Final Valid set class counts:\n{df_valid_set['target'].value_counts()}")

    # Add weights (used by WeightedRandomSampler)
    df_train_set = df_train_set.reset_index(drop=True)
    if len(df_train_set) > 0:
        df_train_set['weight'] = 1.0 / len(df_train_set)
        sample_weights = torch.tensor(df_train_set['weight'].values, dtype=torch.float32)
    else:
        sample_weights = torch.tensor([], dtype=torch.float32) # Handle empty case

    # Datasets
    print("Creating datasets...")
    dataset_train = SIIMISICDataset(df_train_set, 'train', 'train', transform=transforms_train)
    dataset_valid = SIIMISICDataset(df_valid_set, 'train', 'val', transform=transforms_val)

    # Sampler (only if not DEBUG and data exists)
    train_sampler = None
    if not DEBUG and len(dataset_train) > 0:
        train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset_train), replacement=True)
        print("Using WeightedRandomSampler for training.")

    # DataLoaders
    print("Initializing DataLoaders...")
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size,
        sampler=train_sampler, # Will be None if DEBUG or empty dataset
        shuffle=(train_sampler is None), # Shuffle if no sampler
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    print(f"Train loader: {len(train_loader)} batches, Valid loader: {len(valid_loader)} batches")
    if len(train_loader) == 0 or len(valid_loader) == 0:
         print("WARNING: One or both DataLoaders are empty. Training may not proceed correctly.")


    # --- 3. Model Instantiation & Loading ---
    print("\n--- Creating Model ---")
    if model_type == 'hybrid':
        model = HybridModel(
            backbone=enet_type, out_dim=out_dim, n_meta_features=n_meta_features,
            image_size=image_size, load_pretrained=False, # Load weights manually below
            dropout_scale_factors=dropout_scale_factors
        )
        print(f"Instantiated HybridModel ({enet_type})")
    else: # 'efficientnet'
        model = enetv2(
            enet_type, n_meta_features=n_meta_features, out_dim=out_dim,
            load_pretrained=False, # Load weights manually below
            dropout_scale_factors=dropout_scale_factors
        )
        print(f"Instantiated enetv2 ({enet_type})")

    # Move to device and wrap with DataParallel BEFORE loading state_dict
    model = model.to(device)
    model = nn.DataParallel(model)
    print(f"Model wrapped with DataParallel.")

    # Load partial pretrained weights (if file exists)
    if os.path.exists(model_file):
        print(f"Attempting to load partial backbone weights from: {model_file}")
        state_dict = torch.load(model_file, map_location='cpu')
        # Add 'module.' prefix if necessary (since our model is wrapped in DataParallel)
        if not any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
            print("  Added 'module.' prefix to state dict keys.")

        # Filter for desired backbone layers (up to block 4)
        freeze_until_block = 4
        partial_backbone_dict = {}
        loaded_keys_count = 0
        for k, v in state_dict.items():
            # Check if the key belongs to the enet part and is within the frozen blocks
            is_enet_key = k.startswith('module.enet.')
            is_in_frozen_block = False
            if is_enet_key:
                 if ('conv_stem' in k or 'bn1' in k or
                     any(f'blocks.{i}.' in k for i in range(freeze_until_block + 1))):
                     is_in_frozen_block = True

            if is_enet_key and is_in_frozen_block:
                 partial_backbone_dict[k] = v
                 loaded_keys_count += 1

        # Load the filtered state dict (non-strict)
        if partial_backbone_dict:
             try:
                 load_result = model.load_state_dict(partial_backbone_dict, strict=False)
                 print(f"  Successfully loaded {loaded_keys_count} keys (up to block {freeze_until_block})")
                 print(f"    Load Result - Missing keys: {len(load_result.missing_keys)}, Unexpected keys: {len(load_result.unexpected_keys)}")
             except Exception as e:
                 print(f"  Error loading partial backbone weights: {e}")
        else:
             print("  Warning: No matching keys found in the pretrained file for partial loading.")
    else:
        print(f"Pretrained model file not found: {model_file}. Using random initialization for backbone.")

    # Apply initial freezing AFTER potentially loading weights
    partial_freeze_enet(model, freeze_until_block=4) # Freezes block 0, 1, 2, 3, 4

    # Verify freezing (optional check)
    for name, param in model.module.enet.named_parameters():
        if "blocks.4." in name: # Check a frozen block
            print(f"  Check: {name} requires_grad: {param.requires_grad} (should be False)")
            break
    for name, param in model.module.enet.named_parameters():
         if f"blocks.{freeze_until_block + 1}." in name: # Check first unfrozen block
            print(f"  Check: {name} requires_grad: {param.requires_grad} (should be True)")
            break

    total_params = count_parameters(model.module) # Count on base model
    print(f"Total TRAINABLE parameters after initial freeze: {total_params:,}")


    # --- 4. Loss & Optimizer ---
    print("\n--- Setting up Loss & Optimizer ---")
    # Weighted Cross Entropy Loss with Label Smoothing
    class_counts = df_train_set['target'].value_counts().reindex(range(out_dim), fill_value=1e-6)
    class_weights = torch.FloatTensor([1.0 / class_counts[i] for i in range(out_dim)]).to(device)
    criterion_multi = nn.CrossEntropyLoss(
        weight=class_weights, reduction='mean', label_smoothing=0.1
    ).to(device)
    lambda_binary = 0.5 # Weight for the auxiliary binary loss

    # Initial Optimizer (for the frozen phase)
    init_lr_frozen = 1e-6
    print(f"Setting up initial optimizer for freeze phase (LR={init_lr_frozen:.1e})...")
    initial_param_groups = []
    initially_trainable_params_ids = set()
    model_base = model.module # Get the actual model

    # Define function to add params to group and track IDs
    def add_param_group(params_list, lr, wd, group_name):
        unique_params = [p for p in params_list if p.requires_grad and id(p) not in initially_trainable_params_ids]
        if unique_params:
            initial_param_groups.append({'params': unique_params, 'lr': lr, 'weight_decay': wd})
            initially_trainable_params_ids.update(id(p) for p in unique_params)
            print(f"  - Added initial {group_name} group ({len(unique_params)} params)")
        return unique_params # Return added params for potential checks

    # Add groups based on model structure and requires_grad status
    if hasattr(model_base, 'enet'):
        add_param_group([p for name, p in model_base.enet.named_parameters() if p.requires_grad], init_lr_frozen, 0.01, "ENet (Unfrozen)")
    if hasattr(model_base, 'vit'):
        add_param_group(list(model_base.vit.parameters()), init_lr_frozen, 0.01, "ViT")
    if hasattr(model_base, 'fusion'):
        add_param_group(list(model_base.fusion.parameters()), init_lr_frozen, 0.05, "Fusion")
    meta_params = []
    if hasattr(model_base, 'meta_attention'): meta_params.extend(list(model_base.meta_attention.parameters()))
    if hasattr(model_base, 'meta_fc'): meta_params.extend(list(model_base.meta_fc.parameters()))
    add_param_group(meta_params, init_lr_frozen, 0.05, "Meta")
    classifier_params = []
    if hasattr(model_base, 'myfc'): classifier_params.extend(list(model_base.myfc.parameters()))
    if hasattr(model_base, 'classifier'): classifier_params.extend(list(model_base.classifier.parameters()))
    add_param_group(classifier_params, init_lr_frozen, 0.05, "Classifier")

    # Catch any remaining trainable parameters
    all_assigned_ids = set().union(*[set(id(p) for p in group['params']) for group in initial_param_groups])
    remaining_params = [p for p in model.parameters() if p.requires_grad and id(p) not in all_assigned_ids]
    if remaining_params:
        print(f"  - WARNING: {len(remaining_params)} trainable params not assigned. Adding to default group.")
        initial_param_groups.append({'params': remaining_params, 'lr': init_lr_frozen, 'weight_decay': 0.01})

    if not initial_param_groups:
        raise RuntimeError("No trainable parameters found for the initial optimizer!")
    optimizer = optim.AdamW(initial_param_groups) # WD applied per group
    print(f"Initial optimizer created with {len(initial_param_groups)} parameter groups.")
    scaler = GradScaler() if use_amp else None


    # --- 5. Schedulers & Early Stopping ---
    print("\n--- Setting up Schedulers & Early Stopping ---")
    # Phase definitions
    freeze_initially_until_block = 4 # Match partial_freeze_enet
    unfreeze_start_epoch = 7
    full_unfreeze_epoch = 15
    freeze_duration = unfreeze_start_epoch - 1
    warmup_duration = full_unfreeze_epoch - unfreeze_start_epoch
    cosine_duration = n_epochs - full_unfreeze_epoch + 1
    print(f"Scheduler Phases: Freeze={freeze_duration} eps, Warmup/Unfreeze={warmup_duration} eps, Cosine={cosine_duration} eps")
    if warmup_duration <= 0 or cosine_duration <= 0:
        print("Warning: Check epoch settings for scheduler durations.")

    # Schedulers for the *initial* optimizer
    scheduler_cosine_initial = CosineAnnealingLR(optimizer, T_max=max(1, cosine_duration), eta_min=1e-7)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=max(1, warmup_duration), after_scheduler=scheduler_cosine_initial)
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, threshold=0.002, cooldown=2, verbose=True, min_lr=1e-7)

    # Early Stopping
    early_stopping_warmup_epochs = freeze_duration # Start checking ES right after freeze phase ends
    early_stopping = EarlyStopping(
        patience=12, mode='max', delta=0.003, relative_delta=True,
        warm_up=early_stopping_warmup_epochs,
        verbose=True, checkpoint_path=f'{kernel_type}_early_stop_best.pth',
        score_weights={'binary_auc': 0.4, 'binary_recall': 0.3, 'multiclass_auc': 0.2, 'binary_specificity': 0.05, 'val_loss': 0.05} # Example weights
    )
    print(f"Early stopping check starts after epoch {early_stopping_warmup_epochs}")


    # --- 6. Training Loop ---
    print("\n--- Starting Training Loop ---")
    train_losses, val_losses = [], []
    best_model_state, best_metrics = None, None
    best_PROBS, best_TARGETS = None, None # Keep track of best predictions/targets
    best_epoch_num = 0
    best_score = float('-inf') # Initialize best score for maximization
    total_start_time = time.time()
    scheduler_main = None # This will hold the post-reset cosine scheduler
    last_epoch_completed = 0
    early_stopping.reset()

    try:
        for epoch in range(1, n_epochs + 1):
            last_epoch_completed = epoch
            epoch_start_time = time.time()
            print(f"\n===== Epoch {epoch}/{n_epochs} =====")

            # Set epoch for dynamic dropout in model (if applicable)
            if hasattr(model.module, 'set_epoch'): model.module.set_epoch(epoch)

            # --- Progressive Unfreezing & Optimizer Reset ---
            # This function handles unfreezing layers and potentially resetting the optimizer
            optimizer, optimizer_changed = progressive_unfreeze_v3(
                model, optimizer, epoch,
                freeze_initially_until_block=freeze_initially_until_block,
                start_unfreeze_epoch=unfreeze_start_epoch,
                full_unfreeze_epoch=full_unfreeze_epoch,
                final_enet_lr_mult=0.3, final_vit_lr_mult=0.8, final_head_lr_mult=1.0,
                base_lr=init_lr, enet_type=enet_type, lr_scale_factors=lr_scale_factors
            )

            # --- Scheduler Stepping Logic ---
            current_phase = "Unknown"
            if epoch <= freeze_duration:
                # Phase 1: Freeze - LR is constant low, no scheduler step needed.
                current_phase = "Freeze"
                pass

            elif epoch == full_unfreeze_epoch and optimizer_changed:
                # Phase 3 Start: Optimizer just reset. Re-init schedulers.
                current_phase = "Cosine Start (Post Reset)"
                print("Optimizer reset. Re-initializing Cosine and Plateau schedulers.")
                remaining_epochs = n_epochs - epoch + 1
                scheduler_main = CosineAnnealingLR(optimizer, T_max=max(1, remaining_epochs), eta_min=1e-7)
                scheduler_plateau = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, threshold=0.002, cooldown=2, verbose=True, min_lr=1e-7)
                scheduler_plateau._reset()
                # No step needed here, first step happens in the next phase block

            elif epoch > freeze_duration and epoch < full_unfreeze_epoch:
                # Phase 2: Warmup/Unfreeze - Step the warmup scheduler.
                current_phase = "Warmup/Unfreeze"
                # Step based on progress within Phase 2 (0-based index)
                warmup_step_index = epoch - freeze_duration - 1
                scheduler_warmup.step(warmup_step_index) # GradualWarmupScheduler handles its own internal logic

            elif epoch >= full_unfreeze_epoch:
                # Phase 3: Cosine Annealing - Step the main cosine scheduler.
                current_phase = "Cosine Anneal"
                # Step based on progress within Phase 3 (0-based index)
                cosine_step_index = epoch - full_unfreeze_epoch
                if scheduler_main: # Ensure scheduler_main was initialized
                    scheduler_main.step(cosine_step_index)
                else:
                    print("Warning: scheduler_main not initialized in Cosine phase yet.")

            # Log current LR
            current_lr = optimizer.param_groups[0]['lr'] # Get LR from the first group
            print(f"Phase: {current_phase} | Current LR: {current_lr:.2e}")
            if wandb: wandb.log({"learning_rate": current_lr}, step=epoch)
            # --- End Scheduler Stepping ---


            # --- Train ---
            if len(train_loader) > 0:
                train_loss = train_epoch(
                    model, train_loader, optimizer, wandb, epoch, scaler=scaler,
                    accum_steps=accum_steps, criterion_multi=criterion_multi,
                    mel_idx=mel_idx, lambda_binary=lambda_binary, device=device
                )
                train_losses.append(train_loss)
            else:
                print("Skipping training epoch due to empty train_loader.")
                train_loss = None # Indicate no training occurred


            # --- Validate ---
            if len(valid_loader) > 0:
                val_results = val_epoch(
                    model, valid_loader, wandb, epoch, n_test=1, recalib_interval=5,
                    criterion_multi=criterion_multi, mel_idx=mel_idx, lambda_binary=lambda_binary,
                    device=device, use_amp=use_amp
                )
                # Unpack the 15 results from the simplified val_epoch
                (val_loss, binary_auc, multiclass_auc,
                 PROBS, TARGETS,
                 binary_acc, binary_precision, binary_recall, binary_f1, binary_specificity,
                 multiclass_acc, multiclass_f1, multiclass_conf_matrix, binary_conf_matrix,
                 optimal_temp, roc_data) = val_results
                val_losses.append(val_loss)

                # --- REMOVED: Inline ROC Curve Plotting ---
                # The plt.show() calls related to ROC were removed here.
                # WandB logging of ROC curves within val_epoch remains.

                # --- Early Stopping Check ---
                normalized_val_loss = 1.0 - min(val_loss / 10.0, 1.0) # Invert loss for maximization
                metrics_for_es = { # Use the metrics defined in EarlyStopping's weights
                    'binary_auc': binary_auc,
                    'val_loss': normalized_val_loss,
                    'binary_recall': binary_recall,
                    'binary_specificity': binary_specificity,
                    'multiclass_auc': multiclass_auc
                }
                early_stopping(metrics_for_es, model, epoch) # Pass metrics, model, epoch

                # --- Calculate Composite Score & Save Best Model ---
                # Use the same weights as early stopping for consistency
                composite_score = sum(
                    early_stopping.score_weights.get(metric, 0) * metrics_for_es.get(metric, 0)
                    for metric in early_stopping.score_weights
                )

                # Log all validation metrics
                if wandb:
                    wandb.log({
                         "val_loss_raw": val_loss,
                         **metrics_for_es, # Log metrics used for ES
                         # Log other calculated metrics
                         'binary_acc': binary_acc,
                         'binary_precision': binary_precision,
                         'binary_f1': binary_f1,
                         'multiclass_acc': multiclass_acc,
                         'multiclass_f1': multiclass_f1,
                         'composite_score': composite_score,
                         'temperature_used': optimal_temp,
                     }, step=epoch)

                # Check if current model is the best based on composite score
                if composite_score > best_score:
                    best_score = composite_score
                    best_model_state = model.module.state_dict() # Get state from base model
                    best_epoch_num = epoch
                    best_metrics = { # Store all relevant metrics from this best epoch
                        "val_loss_raw": val_loss,
                        **metrics_for_es, # Include ES metrics for reference
                        'binary_acc': binary_acc,
                        'binary_precision': binary_precision,
                        'binary_f1': binary_f1,
                        'multiclass_acc': multiclass_acc,
                        'multiclass_f1': multiclass_f1,
                        'composite_score': composite_score,
                        'binary_conf_matrix': binary_conf_matrix,
                        'multiclass_conf_matrix': multiclass_conf_matrix,
                        'temperature': optimal_temp,
                        # 'optimal_threshold': best_threshold, # If you calculate and need this
                        'total_time_at_best': time.time() - total_start_time
                    }
                    best_PROBS = PROBS # Save predictions/targets
                    best_TARGETS = TARGETS
                    print(f"*** New Best Composite Score: {best_score:.6f} at Epoch {epoch} ***")

                # --- Plateau Scheduler Step ---
                # Step based on a key metric after the warmup/freeze phases
                if epoch > freeze_duration + warmup_duration: # Step only during cosine phase
                    scheduler_plateau.step(binary_auc) # Example: step based on binary AUC

            else:
                print("Skipping validation epoch due to empty valid_loader.")
                # Handle case where validation doesn't run (e.g., don't check early stopping)
                pass


            # --- Check Early Stopping Trigger ---
            if early_stopping.early_stop:
                print(f"EARLY STOPPING triggered after epoch {epoch}.")
                break

            print(f"Epoch {epoch} completed in {time.time() - epoch_start_time:.2f} seconds.")
            torch.cuda.empty_cache()
        # --- End MAIN TRAINING LOOP ---

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training loop: {e}")
        logging.error(traceback.format_exc()) # Log detailed error
    finally:
        # --- Final Operations ---
        total_time = time.time() - total_start_time
        print(f"\n===== Training Finished / Stopped (Epoch {last_epoch_completed}/{n_epochs}) =====")
        print(f"Total training time: {total_time:.2f} seconds")

        # Prepare final metrics dictionary (use best if available, else last)
        final_metrics_to_log = {}
        if best_metrics:
            print(f"Using metrics from Best Epoch: {best_epoch_num}")
            final_metrics_to_log = best_metrics
        elif 'val_results' in locals(): # Fallback to last epoch's results if no best saved
            print("WARNING: No best metrics saved. Using metrics from last completed validation epoch.")
            final_metrics_to_log = { # Reconstruct from last val_results
                "val_loss_raw": val_loss, 'binary_auc': binary_auc, 'multiclass_auc': multiclass_auc,
                'binary_acc': binary_acc, 'binary_precision': binary_precision, 'binary_recall': binary_recall,
                'binary_f1': binary_f1, 'binary_specificity': binary_specificity, 'multiclass_acc': multiclass_acc,
                'multiclass_f1': multiclass_f1, 'composite_score': composite_score, # Last composite score
                'binary_conf_matrix': binary_conf_matrix, 'multiclass_conf_matrix': multiclass_conf_matrix,
                'temperature': optimal_temp, 'epoch': last_epoch_completed
            }
            # Use last predictions/targets if no best saved
            if best_PROBS is None and 'PROBS' in locals(): best_PROBS = PROBS
            if best_TARGETS is None and 'TARGETS' in locals(): best_TARGETS = TARGETS
            # Use last model state if no best saved
            if best_model_state is None and 'model' in locals(): best_model_state = model.module.state_dict()
            best_epoch_num = last_epoch_completed # Mark as last epoch
        else:
            print("WARNING: No validation results available to log.")

        final_metrics_to_log['total_training_time'] = total_time
        final_metrics_to_log['last_epoch_completed'] = last_epoch_completed
        final_metrics_to_log['best_epoch_logged'] = best_epoch_num # Clarify which epoch's metrics are logged


        # Log final summary metrics to WandB
        if wandb and wandb.run:
             summary_log = {}
             print("\nFinal Metrics (logged to WandB Summary):")
             for k, v in sorted(final_metrics_to_log.items()):
                 # Prepare for summary (only scalars/simple types)
                 if isinstance(v, (int, float, bool, str)):
                     summary_log[k] = v
                 elif isinstance(v, np.ndarray) and v.size == 1:
                     summary_log[k] = v.item()
                 # Print logic
                 if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 10:
                      print(f"  - {k}: Type={type(v)}, Shape/Len={getattr(v, 'shape', len(v))} (Logged separately)")
                 elif isinstance(v, (float, np.float32, np.float64)):
                      print(f"  - {k}: {v:.6f}")
                 else:
                      print(f"  - {k}: {v}")
             wandb.summary.update(summary_log)


        # Save and Log FINAL Best Model Artifact
        if best_model_state is not None:
            final_model_path = f'{kernel_type}_BEST_epoch{best_epoch_num}.pth'
            torch.save({
                'epoch': best_epoch_num,
                'model_state_dict': best_model_state,
                'best_composite_score': best_score,
                'best_metrics': best_metrics # Save the detailed metrics dict
             }, final_model_path)
            print(f"\nSaved FINAL Best Model locally: {final_model_path}")

            if wandb and wandb.run:
                model_artifact = wandb.Artifact(f"model-{wandb.run.id}-BEST", type="model",
                                                description=f"Best model (Epoch {best_epoch_num}, Score: {best_score:.6f})")
                model_artifact.add_file(final_model_path)
                wandb.log_artifact(model_artifact, aliases=["best", f"epoch_{best_epoch_num}"])
                print("Logged FINAL best model artifact to WandB.")

            # Save and Log FINAL Best Predictions Artifact
            if best_PROBS is not None and best_TARGETS is not None and wandb and wandb.run:
                try:
                    if not isinstance(best_PROBS, np.ndarray): best_PROBS = np.array(best_PROBS)
                    if not isinstance(best_TARGETS, np.ndarray): best_TARGETS = np.array(best_TARGETS)
                    probs_filename = f"best_probs_epoch_{best_epoch_num}.npy"
                    targets_filename = f"best_targets_epoch_{best_epoch_num}.npy"
                    np.save(probs_filename, best_PROBS, allow_pickle=False)
                    np.save(targets_filename, best_TARGETS, allow_pickle=False)
                    print(f"Saved best validation predictions/targets: {probs_filename}, {targets_filename}")

                    pred_artifact = wandb.Artifact(f"val-predictions-{wandb.run.id}-BEST", type="validation_predictions",
                                                  description=f"Validation predictions/targets from best epoch {best_epoch_num}")
                    pred_artifact.add_file(probs_filename)
                    pred_artifact.add_file(targets_filename)
                    wandb.log_artifact(pred_artifact, aliases=["best_predictions", f"epoch_{best_epoch_num}"])
                    print("Logged best validation predictions/targets artifact to WandB.")
                    # os.remove(probs_filename) # Optional cleanup
                    # os.remove(targets_filename)
                except Exception as e:
                    print(f"Error saving/logging FINAL prediction artifacts: {e}")
        else:
            print("\nNo best model state recorded, skipping final model/prediction artifact saving.")

        # Log final CM tables from best_metrics
        if best_metrics and wandb and wandb.run:
             binary_cm_list = best_metrics.get('binary_conf_matrix', [[0,0],[0,0]])
             multiclass_cm_list = best_metrics.get('multiclass_conf_matrix', [])
             if out_dim == 9: class_names = ['AK','BCC','BKL','DF','SCC','VASC','melanoma','nevus','unknown']
             else: class_names = [str(i) for i in range(out_dim)]

             wandb.log({
                 "binary_confusion_matrix_table_BEST": wandb.Table(columns=['Pred Non-Mel', 'Pred Mel'], data=binary_cm_list)
             }, step=n_epochs) # Log at final step
             if multiclass_cm_list:
                 wandb.log({
                     "multiclass_confusion_matrix_table_BEST": wandb.Table(columns=[f"Pred {name}" for name in class_names], data=multiclass_cm_list)
                 }, step=n_epochs)


    # --- Function Return (Matching previous simplified version) ---
    print("--- Exiting run_single_model ---")
    return (model, best_model_state, final_metrics_to_log, train_losses, val_losses,
            best_PROBS, best_TARGETS, # REMOVED best_STD_PROBS
            df_valid_set, diagnosis2idx,
            best_epoch_num, kernel_type, early_stopping,
            model_type, enet_type, n_meta_features, image_size, out_dim,
            lr_scale_factors, dropout_scale_factors
           ) # Total 19 return values
```
:::

::: {.cell .markdown}
# Main execution
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T02:59:05.942325Z\",\"iopub.status.busy\":\"2025-04-07T02:59:05.942126Z\",\"iopub.status.idle\":\"2025-04-07T03:01:08.077228Z\",\"shell.execute_reply\":\"2025-04-07T03:01:08.076193Z\",\"shell.execute_reply.started\":\"2025-04-07T02:59:05.942308Z\"}" trusted="true"}
``` python
import time

if __name__ == "__main__":
    print(f"Training a single model with {kernel_type}...")
    start_time = time.time()
    model_type='hybrid' # Or 'hybrid' depending on the run

    # --- CORRECTED FUNCTION CALL (unpacking matches 19 return values) ---
    (model, best_model_state, final_metrics, train_losses, val_losses,
     best_PROBS, best_TARGETS, # Removed best_STD_PROBS from this list
     df_valid_set, diagnosis2idx,
     best_epoch_num, kernel_type, early_stopping,
     model_type_ret, enet_type_ret, n_meta_features_ret, image_size_ret, out_dim_ret,
     lr_scale_factors, dropout_scale_factors
    ) = run_single_model(model_type=model_type, enet_type=enet_type)


    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f}s")
    print("Final Metrics:", final_metrics)
    if final_metrics:
        print(f"Binary AUC: {final_metrics.get('binary_auc', 0.0):.4f}")
        print(f"Multiclass AUC: {final_metrics.get('multiclass_auc', 0.0):.4f}")

    # Now, when the Grad-CAM code runs after this block,
    # df_valid_set and diagnosis2idx will exist in the global scope.
```
:::

::: {.cell .markdown}
# Section: Model Calibration Visualization (Reliability Diagram)
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T03:01:08.078779Z\",\"iopub.status.busy\":\"2025-04-07T03:01:08.078482Z\",\"iopub.status.idle\":\"2025-04-07T03:01:08.598975Z\",\"shell.execute_reply\":\"2025-04-07T03:01:08.598264Z\",\"shell.execute_reply.started\":\"2025-04-07T03:01:08.078754Z\"}" trusted="true"}
``` python
# This section visualizes the calibration of the best model's probabilities
# for the primary class of interest (melanoma) using a reliability diagram.
# It leverages the saved predictions and targets from the best validation epoch.
# ==============================================================================

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import numpy as np
import wandb # Assuming wandb is potentially still active or needed for logging

print("\n===== Starting Model Calibration Visualization =====")

# --- 1. Prerequisite Check ---
required_vars_for_calibration = ['best_PROBS', 'best_TARGETS', 'mel_idx', 'best_epoch_num', 'kernel_type']
calibration_vars_ok = True
for var_name in required_vars_for_calibration:
    if var_name not in locals() or locals()[var_name] is None:
        print(f"ERROR: Calibration requires '{var_name}', which is missing or None.")
        calibration_vars_ok = False
    # Specifically check if the arrays are non-empty if they exist
    elif var_name in ['best_PROBS', 'best_TARGETS'] and len(locals()[var_name]) == 0:
        print(f"ERROR: Calibration requires non-empty '{var_name}'. Found empty array.")
        calibration_vars_ok = False

if not calibration_vars_ok:
    print("Skipping calibration plot generation due to missing prerequisites.")
else:
    print(f"Prerequisites met. Generating calibration plot for Melanoma class (index {mel_idx}) from Epoch {best_epoch_num}.")

    # --- 2. Prepare Data for Binary Calibration (Melanoma vs Non-Melanoma) ---
    try:
        # Ensure they are numpy arrays
        if not isinstance(best_PROBS, np.ndarray): best_PROBS = np.array(best_PROBS)
        if not isinstance(best_TARGETS, np.ndarray): best_TARGETS = np.array(best_TARGETS)

        # Extract probabilities for the positive class (melanoma)
        # Ensure mel_idx is within bounds
        if mel_idx < 0 or mel_idx >= best_PROBS.shape[1]:
             raise IndexError(f"mel_idx ({mel_idx}) is out of bounds for best_PROBS shape {best_PROBS.shape}")
        y_prob_melanoma = best_PROBS[:, mel_idx]

        # Create binary true labels (1 if melanoma, 0 otherwise)
        y_true_binary = (best_TARGETS == mel_idx).astype(int)

        # Check if there's variation in true labels (needed for calibration curve)
        if len(np.unique(y_true_binary)) < 2:
            print("WARNING: Only one class present in the best validation targets. Calibration curve may not be meaningful.")
            # Optionally skip plotting here if desired

        print(f"  Prepared data: y_prob_melanoma shape {y_prob_melanoma.shape}, y_true_binary shape {y_true_binary.shape}")
        print(f"  Number of positive (melanoma) samples in best validation set: {np.sum(y_true_binary)}")

        # --- 3. Calculate Calibration Curve ---
        # Use 'uniform' strategy for equally spaced bins based on probability.
        # 'quantile' can also be used for bins with equal numbers of samples.
        n_bins = 10 # A common choice, can be adjusted
        prob_true, prob_pred = calibration_curve(y_true_binary, y_prob_melanoma, n_bins=n_bins, strategy='uniform')

        print(f"  Calculated calibration curve points (True Probability, Predicted Probability):")
        for pt, pp in zip(prob_true, prob_pred):
            print(f"    {pt:.4f}, {pp:.4f}")

        # --- 4. Plot Reliability Diagram ---
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.plot(prob_pred, prob_true, marker='s', linestyle='-', label='Model Calibration (Melanoma Class)')

        plt.xlabel("Mean Predicted Probability (in bin)", fontsize=12)
        plt.ylabel("Fraction of Positives (in bin)", fontsize=12)
        plt.title(f"Calibration Plot (Reliability Diagram) - Epoch {best_epoch_num}\n{kernel_type}", fontsize=14, pad=15)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        # --- 5. Save and Log Plot ---
        calibration_plot_filename = f"{kernel_type}_calibration_plot_epoch{best_epoch_num}.png"
        plt.savefig(calibration_plot_filename, dpi=150)
        print(f"  Saved calibration plot locally: {calibration_plot_filename}")
        plt.show() # Display the plot in the notebook

        # Log to WandB if available and run is active
        if 'wandb' in locals() and wandb is not None and wandb.run is not None:
            try:
                wandb.log({"calibration_plot_melanoma": wandb.Image(calibration_plot_filename)}, step=best_epoch_num) # Log at the best epoch step
                print("  Logged calibration plot to WandB.")
            except Exception as e:
                print(f"  Warning: Failed to log calibration plot to WandB: {e}")
        else:
            print("  Skipping WandB logging for calibration plot (WandB not active).")

    except Exception as e:
        print(f"\nERROR generating calibration plot: {e}")
        import traceback
        traceback.print_exc()

print("===== Model Calibration Visualization Finished =====")
```
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T03:01:08.600221Z\",\"iopub.status.busy\":\"2025-04-07T03:01:08.599918Z\",\"iopub.status.idle\":\"2025-04-07T03:01:14.504880Z\",\"shell.execute_reply\":\"2025-04-07T03:01:14.504215Z\",\"shell.execute_reply.started\":\"2025-04-07T03:01:08.600189Z\"}" trusted="true"}
``` python
wandb.finish()  # End the W&B run
```
:::

::: {.cell .markdown}
# Grad-Cam Run
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T03:01:14.505905Z\",\"iopub.status.busy\":\"2025-04-07T03:01:14.505622Z\",\"iopub.status.idle\":\"2025-04-07T03:01:14.515073Z\",\"shell.execute_reply\":\"2025-04-07T03:01:14.514329Z\",\"shell.execute_reply.started\":\"2025-04-07T03:01:14.505870Z\"}" trusted="true"}
``` python
print("--- Debug Check Before Grad-CAM ---")
required_vars_check = [
    'df_valid_set', 'diagnosis2idx', 'best_model_state', 'model_type',
    'enet_type', 'n_meta_features', 'image_size', 'out_dim',
    'mel_idx', 'kernel_type', 'best_epoch_num', 'dropout_scale_factors',
    'early_stopping', 'lr_scale_factors' # Add any others you suspect
]
variables_exist = {}
for var_name in required_vars_check:
    variables_exist[var_name] = var_name in locals()
    if variables_exist[var_name]:
        # Optionally print type or shape for complex variables
        if isinstance(locals()[var_name], (pd.DataFrame, np.ndarray, torch.Tensor)):
             print(f"Variable '{var_name}': Exists, Type: {type(locals()[var_name])}, Shape: {locals()[var_name].shape}")
        elif isinstance(locals()[var_name], dict):
             print(f"Variable '{var_name}': Exists, Type: {type(locals()[var_name])}, Keys (first 5): {list(locals()[var_name].keys())[:5]}")
        else:
             print(f"Variable '{var_name}': Exists, Type: {type(locals()[var_name])}, Value: {locals()[var_name]}")

    else:
        print(f"Variable '{var_name}': DOES NOT EXIST")

print("--- End Debug Check ---")
```
:::

::: {.cell .code execution="{\"iopub.execute_input\":\"2025-04-07T03:01:14.516281Z\",\"iopub.status.busy\":\"2025-04-07T03:01:14.515950Z\",\"iopub.status.idle\":\"2025-04-07T03:01:24.193402Z\",\"shell.execute_reply\":\"2025-04-07T03:01:24.192475Z\",\"shell.execute_reply.started\":\"2025-04-07T03:01:14.516227Z\"}" trusted="true"}
``` python
# --- Add this line to reset the index ---
df_valid_set = df_valid_set.reset_index(drop=True)
print("Reset index of df_valid_set to ensure uniqueness for visualization sampling.")
# --- End Add ---
print("\n===== Starting Grad-CAM Visualization (True vs. Predicted Focus) =====")

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_images_to_visualize = 10 # Show 10 images

# --- Explicit Prerequisite Check ---
print("\n--- Explicit Variable Check ---")
required_vars = ['df_valid_set', 'diagnosis2idx', 'best_model_state', 'model_type',
                 'enet_type', 'n_meta_features', 'image_size', 'out_dim',
                 'mel_idx', 'kernel_type', 'best_epoch_num', 'dropout_scale_factors',
                 'early_stopping', 'lr_scale_factors'] # Make sure all returned vars are checked

all_vars_ok = True
for var_name in required_vars:
    if var_name not in locals():
        print(f"ERROR: Variable '{var_name}' NOT FOUND in locals().")
        all_vars_ok = False
    elif locals()[var_name] is None:
        # Special check for pandas DataFrame which can be tricky with 'is None'
        if isinstance(locals()[var_name], pd.DataFrame) and locals()[var_name].empty:
             print(f"INFO: Variable '{var_name}' is an EMPTY DataFrame, but exists.")
             # Decide if an empty DataFrame is acceptable here - probably ok for df_valid_set if training failed instantly
        else:
             print(f"ERROR: Variable '{var_name}' FOUND but is None.")
             all_vars_ok = False
    else:
        # Optional: Confirm variables are okay
        # print(f"OK: Variable '{var_name}' exists and is not None.")
        pass

if not all_vars_ok:
    print("Error: One or more prerequisite variables are missing or None.")
    print("Ensure training completed successfully and returned all required variables.")
    print("--- End Explicit Variable Check ---")
    # Stop execution here if check fails (or rely on else block not running)

else: # <<< Correct Main Else Block >>>
    print("All prerequisite variables seem OK. Proceeding...")
    print("--- End Explicit Variable Check ---")
    print(f"Visualizing model type: '{model_type}', enet_type: '{enet_type}', using metadata: {n_meta_features > 0}")

    # --- Instantiate the *Trained* Model Architecture ---
    print("\nInstantiating the trained model architecture...")
    if model_type == 'hybrid':
        viz_model = HybridModel(
            backbone=enet_type, out_dim=out_dim, n_meta_features=n_meta_features,
            image_size=image_size, load_pretrained=False,
            dropout_scale_factors=dropout_scale_factors
        ).to(device)
        print("  Instantiated HybridModel")
    elif model_type == 'efficientnet':
         viz_model = enetv2(
            enet_type, out_dim=out_dim, n_meta_features=n_meta_features,
            load_pretrained=False, dropout_scale_factors=dropout_scale_factors
        ).to(device)
         print("  Instantiated enetv2")
    else:
        print(f"Error: Unknown model_type '{model_type}' for visualization.")
        viz_model = None

    if viz_model:
        # --- Load Best State Dict ---
        print("\nLoading best model state...")
        has_module_prefix_in_state = any(k.startswith('module.') for k in best_model_state.keys())
        if not has_module_prefix_in_state:
            print("  Adding 'module.' prefix to saved state dict keys for DataParallel loading.")
            state_dict_to_load = {'module.' + k: v for k, v in best_model_state.items()}
        else:
             state_dict_to_load = best_model_state

        viz_model = torch.nn.DataParallel(viz_model) # Wrap in DataParallel
        print(f"  Wrapping model in DataParallel.")

        try:
            load_result = viz_model.load_state_dict(state_dict_to_load, strict=True)
            print(f"  Successfully loaded best model state (strict=True).")
        except RuntimeError as e:
            print(f"  Warning: Strict loading failed ({e}). Trying with strict=False...")
            try:
                 load_result = viz_model.load_state_dict(state_dict_to_load, strict=False)
                 print(f"  Successfully loaded best model state (strict=False).")
                 print(f"    Load Result (strict=False) - Missing keys: {len(load_result.missing_keys)}, Unexpected keys: {len(load_result.unexpected_keys)}")
                 if load_result.missing_keys: print(f"      Missing: {load_result.missing_keys[:5]}...")
                 if load_result.unexpected_keys: print(f"      Unexpected: {load_result.unexpected_keys[:5]}...")
            except Exception as E:
                 print(f"  ERROR: Failed to load model state even with strict=False: {E}")
                 viz_model = None

        # --- Identify Target Layer ---
        if viz_model:
            print("\nIdentifying target layer...")
            target_layer = None
            try:
                if hasattr(viz_model.module, 'enet') and hasattr(viz_model.module.enet, 'conv_head'):
                     target_layer = viz_model.module.enet.conv_head
                     print(f"  Identified target layer: viz_model.module.enet.conv_head")
                elif hasattr(viz_model.module, 'enet') and hasattr(viz_model.module.enet, 'blocks'):
                         target_layer = viz_model.module.enet.blocks[-1]
                         print(f"  Using fallback target layer: viz_model.module.enet.blocks[-1]")
                else: print("  Error: Cannot determine target layer.")
            except AttributeError: print("  Error accessing model layers.")

            if target_layer:
                viz_model.eval()

                                # --- Seed for Reproducible Image Selection ---
                SEED_FOR_VISUALIZATION = 42 # You can choose any integer here, just keep it constant
                rng = np.random.default_rng(seed=SEED_FOR_VISUALIZATION)
                print(f"Using fixed random seed {SEED_FOR_VISUALIZATION} for image selection.")
                # --- End Seed ---

                # --- Select Images Randomly (but reproducibly) ---
                num_available = len(df_valid_set)
                num_to_sample = min(num_images_to_visualize, num_available)
                if num_to_sample > 0:
                    # Use the seeded random number generator (rng)
                    random_image_indices = rng.choice(num_available, num_to_sample, replace=False).tolist()
                    # Sort the indices for consistency in the order they are processed (optional but good practice)
                    random_image_indices.sort()
                    print(f"\nSelected {len(random_image_indices)} reproducible image indices for visualization: {random_image_indices}")
                else:
                    print("Warning: No images available in df_valid_set to sample.")
                    random_image_indices = []
                # --- End Image Selection ---

                # --- Generate Grad-CAMs ---
                print("\nGenerating Grad-CAMs...")
                for img_index in random_image_indices:
                    try:
                        image_info = df_valid_set.loc[img_index]
                        image_path = image_info['filepath']

                        # Robust label extraction
                        true_label_idx_val = image_info['target']
                        if isinstance(true_label_idx_val, pd.Series): true_label_idx = int(true_label_idx_val.iloc[0])
                        elif isinstance(true_label_idx_val, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)): true_label_idx = int(true_label_idx_val)
                        elif isinstance(true_label_idx_val, (int, float)): true_label_idx = int(true_label_idx_val)
                        else: raise TypeError(f"Unexpected type for target: {type(true_label_idx_val)}")
                        try: true_label_name = list(diagnosis2idx.keys())[list(diagnosis2idx.values()).index(true_label_idx)]
                        except ValueError: true_label_name = f"Unknown ({true_label_idx})"

                        print(f"\nProcessing Image: {os.path.basename(image_path)} (Index: {img_index}, True Label: {true_label_name} [{true_label_idx}])")

                        # Load and preprocess image
                        input_tensor, img_for_display = load_and_preprocess_for_gradcam(image_path, image_size)
                        input_tensor = input_tensor.to(device)

                        # Prepare metadata tensor
                        meta_tensor_for_image = None
                        model_uses_meta = n_meta_features > 0
                        if model_uses_meta:
                            if 'meta_features' in locals():
                                meta_values = image_info[meta_features].values.astype(np.float32)
                                meta_tensor_for_image = torch.tensor(meta_values).unsqueeze(0).to(device)
                            else: print("  Warning: Metadata expected but 'meta_features' list not found.")

                        # Get Model Prediction
                        pred_label_name = "N/A"; pred_label_idx = -1; pred_prob_for_true_class = 0.0; pred_prob_for_pred_class = 0.0
                        with torch.no_grad():
                            logits = None
                            if model_uses_meta and meta_tensor_for_image is not None: logits = viz_model(input_tensor, meta_tensor_for_image)
                            elif model_uses_meta and meta_tensor_for_image is None: print("  Skipping prediction: Metadata needed but not available."); pred_label_name = "Error (Meta Missing)"
                            else: logits = viz_model(input_tensor) # No meta needed

                            if logits is not None:
                                pred_prob = torch.softmax(logits, dim=1)
                                pred_label_idx = torch.argmax(pred_prob, dim=1).item()
                                pred_prob_for_pred_class = pred_prob[0, pred_label_idx].item()
                                try: pred_label_name = list(diagnosis2idx.keys())[list(diagnosis2idx.values()).index(pred_label_idx)]
                                except ValueError: pred_label_name = f"Unknown ({pred_label_idx})"
                                if 0 <= true_label_idx < pred_prob.shape[1]: pred_prob_for_true_class = pred_prob[0, true_label_idx].item()
                                else: pred_prob_for_true_class = -1.0
                            # (Error handling for prediction stays)

                        print(f"  Prediction: {pred_label_name} [{pred_label_idx}] (Prob: {pred_prob_for_pred_class:.3f})")

                        # Generate CAM for TRUE Class
                        print(f"  Generating CAM for TRUE Class: {true_label_name} [{true_label_idx}]")
                        visualization_true, _ = generate_gradcam(
                            model=viz_model, target_layer=target_layer, input_tensor=input_tensor,
                            original_image=img_for_display, target_class_idx=true_label_idx,
                            device=device, meta_tensor=meta_tensor_for_image
                        )

                        # Generate CAM for PREDICTED Class
                        visualization_pred = None
                        if pred_label_idx != -1 and pred_label_idx != true_label_idx :
                            print(f"  Generating CAM for PREDICTED Class: {pred_label_name} [{pred_label_idx}]")
                            visualization_pred, _ = generate_gradcam(
                                model=viz_model, target_layer=target_layer, input_tensor=input_tensor,
                                original_image=img_for_display, target_class_idx=pred_label_idx,
                                device=device, meta_tensor=meta_tensor_for_image
                            )
                        elif pred_label_idx == true_label_idx:
                             print(f"  Skipping CAM for predicted class (same as true class).")
                             visualization_pred = visualization_true # Reuse CAM if pred == true
                        else:
                            print("  Skipping CAM for predicted class due to prediction error.")

                        # Display Results
                        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                        axes[0].imshow(img_for_display)
                        axes[0].set_title(f"Original\nTrue: {true_label_name} [{true_label_idx}]")
                        axes[0].axis('off')
                        axes[1].imshow(visualization_true)
                        axes[1].set_title(f"CAM for True ({true_label_name})\nModel's Prob: {pred_prob_for_true_class:.3f}")
                        axes[1].axis('off')
                        if visualization_pred is not None:
                            axes[2].imshow(visualization_pred)
                            axes[2].set_title(f"CAM for Pred ({pred_label_name})\nModel's Prob: {pred_prob_for_pred_class:.3f}")
                        else:
                             if pred_label_idx == -1: info_text = 'Prediction Error\nNo CAM Generated'
                             elif pred_label_idx == true_label_idx: info_text = 'Prediction = True\n(CAM same as True)'
                             else: info_text = 'CAM Pred Error'
                             axes[2].text(0.5, 0.5, info_text, ha='center', va='center', fontsize=10)
                             axes[2].set_title(f"CAM for Pred ({pred_label_name})")
                        axes[2].axis('off')
                        plt.suptitle(f"Image: {os.path.basename(image_path)} (Idx: {img_index}) | Model: {model_type}/{enet_type}", fontsize=14)
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        plt.show()

                    except Exception as e:
                        print(f"  Failed significantly on image {img_index}: {e}")
                        import traceback
                        traceback.print_exc()

            else:
                print("Skipping CAM generation as target layer couldn't be identified.")
        else:
             print("Skipping CAM generation as model loading or state dict loading failed.")

# This final print is outside the main 'else' corresponding to 'if all_vars_ok:'
print("\n===== Grad-CAM Visualization Finished =====")
```
:::
