import os
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
import timm
from transformers import ViTConfig, ViTModel
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from io import BytesIO
import pandas as pd
import base64
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Constants for Preprocessing ---
# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Metadata normalization constants (from training)
AGE_MEAN = 0.0  # Age was just divided by 90.0 in training
AGE_STD = 1.0   # No standardization was applied
SEX_MAPPING = {'male': 1, 'female': 0}
SEX_MISSING_VALUE = -1

# Temperature scaling (no saved optimal temperature)
OPTIMAL_TEMP = 1.0

# Class mapping (inferred from training)
IDX_TO_DIAGNOSIS_DICT = {
    0: 'AK',
    1: 'BCC', 
    2: 'BKL',
    3: 'DF',
    4: 'SCC',
    5: 'VASC',
    6: 'melanoma',
    7: 'nevus',
    8: 'unknown'
}

# Location columns (hardcoded from manual context)
LOCATION_COLS_LIST = [
    'site_anterior torso', 'site_head/neck', 'site_lateral torso',
    'site_lower extremity', 'site_oral/genital', 'site_palms/soles',
    'site_posterior torso', 'site_torso', 'site_upper extremity', 'site_nan'
]

# Grad-CAM Configuration
USE_GRADCAM = True
TARGET_LAYER_NAME = 'conv_head'  # Will be used to get actual layer object

def preprocess_image(image_bytes):
    """
    Preprocess image bytes to tensor following training preprocessing.
    Args:
        image_bytes: Raw image bytes from request
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Read image
    image = Image.open(BytesIO(image_bytes))
    image = np.array(image)
    
    # Convert to RGB if needed
    if image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Apply transforms matching training
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    transformed = transform(image=image)
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.tensor(transformed['image']).permute(2, 0, 1).unsqueeze(0).float()
    return image_tensor

def preprocess_metadata(data_dict):
    """
    Preprocess metadata following training preprocessing.
    Args:
        data_dict: Dictionary with metadata fields
    Returns:
        torch.Tensor: Preprocessed metadata tensor
    """
    meta_features = []
    
    # 1. Sex encoding
    sex = data_dict.get('sex', '')
    sex_encoded = SEX_MAPPING.get(sex, SEX_MISSING_VALUE)
    meta_features.append(float(sex_encoded))
    
    # 2. Age normalization (divide by 90.0 as in training)
    age = float(data_dict.get('age', 0))
    age_normalized = age / 90.0
    meta_features.append(age_normalized)
    
    # 3. N_images (log transformed)
    n_images = np.log1p(float(data_dict.get('n_images', 1)))
    meta_features.append(n_images)
    
    # 4. Image size (already log transformed)
    image_size = float(data_dict.get('image_size', 0))
    meta_features.append(image_size)
    
    # 5. Location one-hot encoding
    location = data_dict.get('anatom_site_general', '')
    
    # Initialize all location features to 0
    location_features = {
        'site_anterior torso': 0.0,
        'site_head/neck': 0.0,
        'site_lateral torso': 0.0,
        'site_lower extremity': 0.0,
        'site_oral/genital': 0.0,
        'site_palms/soles': 0.0,
        'site_posterior torso': 0.0,
        'site_torso': 0.0,
        'site_upper extremity': 0.0,
        'site_nan': 0.0
    }
    
    # Map the input location to the correct site_ column
    location_mapping = {
        'head/neck': 'site_head/neck',
        'upper extremity': 'site_upper extremity',
        'lower extremity': 'site_lower extremity',
        'torso': 'site_torso',
        'anterior torso': 'site_anterior torso',
        'lateral torso': 'site_lateral torso',
        'posterior torso': 'site_posterior torso',
        'palms/soles': 'site_palms/soles',
        'oral/genital': 'site_oral/genital'
    }
    
    if location in location_mapping:
        location_features[location_mapping[location]] = 1.0
    else:
        location_features['site_nan'] = 1.0
    
    # Add location features in the correct order
    meta_features.extend([location_features[col] for col in LOCATION_COLS_LIST])
    
    # Convert to tensor
    meta_tensor = torch.tensor(meta_features, dtype=torch.float32).unsqueeze(0)
    return meta_tensor

# --- Model Architecture Definition ---
class MetadataAttention(nn.Module):
    def __init__(self, n_meta_features, hidden_dim=256):
        super(MetadataAttention, self).__init__()
        self.query = nn.Linear(n_meta_features, hidden_dim)
        self.key = nn.Linear(n_meta_features, hidden_dim)
        self.value = nn.Linear(n_meta_features, hidden_dim)
        self.scale = nn.Parameter(torch.sqrt(torch.tensor([hidden_dim], dtype=torch.float32)), requires_grad=True)
        self.attention_dropout = nn.Dropout(0.2)
    
    def forward(self, meta):
        Q = self.query(meta)
        K = self.key(meta)
        V = self.value(meta)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = nn.functional.softmax(scores, dim=-1)
        attention = self.attention_dropout(attention)
        weighted_meta = torch.matmul(attention, V)
        return weighted_meta

class HybridModel(nn.Module):
    def __init__(self, backbone, out_dim, n_meta_features=0, load_pretrained=True, image_size=448, dropout_scale_factors=None):
        super(HybridModel, self).__init__()
        self.n_meta_features = n_meta_features
        self.out_dim = out_dim
        self.backbone = backbone
        self.dropout_scale_factors = dropout_scale_factors if dropout_scale_factors is not None else {v: 1.0 for v in range(3, 8)}

        # EfficientNet setup
        self.enet = timm.create_model(backbone, pretrained=load_pretrained)
        self.enet_features = self.enet.num_features
        self.enet.reset_classifier(0, '')
        self.pool = nn.AdaptiveAvgPool2d(1)

        # ViT setup
        self.vit_config = ViTConfig(
            image_size=image_size, patch_size=16, hidden_size=192, num_hidden_layers=4,
            num_attention_heads=3, intermediate_size=768, hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.vit = ViTModel(self.vit_config)
        self.vit_features = self.vit_config.hidden_size

        # Fusion Layer
        fusion_dim = self.enet_features + self.vit_features
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.BatchNorm1d(512), nn.SiLU(),
            nn.Dropout(0.45)
        )

        # Meta Pathway (if used)
        if self.n_meta_features > 0:
            meta_hidden_dim = 256
            self.meta_attention = MetadataAttention(n_meta_features, hidden_dim=meta_hidden_dim // 2)
            self.meta_fc = nn.Sequential(
                nn.Linear(meta_hidden_dim // 2, meta_hidden_dim),
                nn.BatchNorm1d(meta_hidden_dim),
                nn.SiLU(),
                nn.Dropout(p=0.4),
                nn.Linear(meta_hidden_dim, 512),  # Changed to match fusion output
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Dropout(p=0.3)
            )
        
        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(512, out_dim)
        )

        self.current_epoch = 0
        self.gradcam_mode = False
        self.fixed_meta = None

    def forward(self, x, x_meta=None):
        # Extract and fuse CNN + ViT features
        cnn_features = self.enet.forward_features(x)
        cnn_features = self.pool(cnn_features).view(cnn_features.size(0), -1)
        vit_features = self.vit(pixel_values=x).last_hidden_state[:, 0]
        combined = torch.cat((cnn_features, vit_features), dim=1)
        x = self.fusion(combined)

        # Process metadata if available and model supports it
        if self.n_meta_features > 0 and x_meta is not None:
            if self.gradcam_mode and self.fixed_meta is not None:
                meta_attended = self.meta_attention(self.fixed_meta)
            else:
                meta_attended = self.meta_attention(x_meta)
            meta_processed = self.meta_fc(meta_attended)
            x = x + meta_processed

        # Final classification
        output = self.classifier(x)
        output = torch.clamp(output, min=-20, max=20)
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=20, neginf=-20)
        return output

# --- Flask Application ---
app = Flask(__name__)

# Model Configuration
MODEL_PATH = 'F:/Kodingan/melanomaPredict/beneran/effnetb5_384_9c_50epo_ext_BEST_epoch48.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BACKBONE = 'efficientnet_b5'
IMAGE_SIZE = 384
OUT_DIM = 9
N_META_FEATURES = 0  # Set to 0 since model was trained without metadata
USE_METADATA = False  # Flag to control metadata usage

# Initialize model
model = None

def load_model():
    global model
    # Initialize model architecture
    model = HybridModel(
        backbone=BACKBONE,
        out_dim=OUT_DIM,
        n_meta_features=N_META_FEATURES,
        image_size=IMAGE_SIZE,
        load_pretrained=False
    )
    
    try:
        # First attempt: Try loading with safe_globals context manager
        print("Attempting to load with safe_globals...")
        from torch.serialization import safe_globals
        import numpy as np
        with safe_globals(['numpy.core.multiarray.scalar']):
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except Exception as e1:
        print(f"First attempt failed: {str(e1)}")
        try:
            # Second attempt: Try loading with weights_only=False (legacy behavior)
            print("Attempting to load model weights with weights_only=False...")
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        except Exception as e2:
            print(f"Second attempt failed: {str(e2)}")
            try:
                # Final attempt: Try loading with pickle
                print("Attempting final load method...")
                import pickle
                with open(MODEL_PATH, 'rb') as f:
                    checkpoint = torch.load(f, map_location=DEVICE, pickle_module=pickle)
            except Exception as e3:
                print(f"All loading attempts failed. Final error: {str(e3)}")
                raise

    try:
        # Handle nested state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Handle module prefix if needed
        if not any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        
        # Remove unexpected keys
        state_dict = {k: v for k, v in state_dict.items() 
                     if not any(x in k for x in ['epoch', 'best_composite_score', 'best_metrics'])}
        
        # Wrap model in DataParallel
        model = nn.DataParallel(model)
        
        # Load state dict with strict=False
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            print(f"Warning: Missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"Warning: Unexpected keys: {load_result.unexpected_keys}")
        
        # Move to device and set eval mode
        model = model.to(DEVICE)
        model.eval()
        print(f"Model loaded successfully and moved to {DEVICE}")
        
    except Exception as e:
        print(f"Error processing state dict: {e}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictor')
def predictor_page():
    return render_template('predict.html')

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template('about.html', title='About')

@app.route('/skin-info') # New route
def skin_info():
    """Renders the skin cancer information page."""
    return render_template('skin_info.html', title='Skin Cancer Info')

def get_target_layer(model):
    """Get the target layer for Grad-CAM visualization."""
    try:
        # For EfficientNet, we want to use the last convolutional layer
        # This is typically in the last block of the network
        base_model = model.module.enet
        
        # Try different potential target layers in order of preference
        if hasattr(base_model, 'conv_head'):
            print("Using conv_head as target layer")
            return base_model.conv_head
        elif hasattr(base_model, 'blocks'):
            # Get the last conv layer in the last block
            last_block = base_model.blocks[-1]
            if hasattr(last_block, 'conv_pwl'):
                print("Using last block's conv_pwl as target layer")
                return last_block.conv_pwl
            elif hasattr(last_block, 'conv_pw'):
                print("Using last block's conv_pw as target layer")
                return last_block.conv_pw
            else:
                # Try to find any conv layer in the last block
                for name, module in reversed(list(last_block.named_modules())):
                    if isinstance(module, nn.Conv2d):
                        print(f"Using last block's {name} as target layer")
                        return module
        
        # If we can't find a suitable layer, try to find the last conv layer in the entire network
        for name, module in reversed(list(base_model.named_modules())):
            if isinstance(module, nn.Conv2d):
                print(f"Using {name} as target layer")
                return module
                
        raise ValueError("Could not find suitable target layer for Grad-CAM")
        
    except Exception as e:
        print(f"Error getting target layer: {e}")
        return None

def prepare_gradcam_image(image_bytes, target_size):
    """Prepare image for Grad-CAM (resized but not normalized)."""
    image = Image.open(BytesIO(image_bytes))
    image = np.array(image)
    
    if image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize while keeping float32 and 0-1 range
    image = cv2.resize(image, (target_size, target_size))
    image = image.astype(np.float32) / 255.0
    
    return image

def generate_gradcam_visualization(model, target_layer, input_tensor, original_image, 
                                 target_class_idx, device, meta_tensor=None):
    """Generate Grad-CAM visualization."""
    try:
        if meta_tensor is not None:
            # Create a wrapper model that includes metadata handling
            class ModelWrapper(nn.Module):
                def __init__(self, model, meta_data):
                    super().__init__()
                    self.model = model
                    self.meta_data = meta_data
                
                def forward(self, x):
                    return self.model(x, self.meta_data)
            
            model_for_cam = ModelWrapper(model, meta_tensor)
        else:
            model_for_cam = model

        if target_layer is None:
            print("No target layer found for Grad-CAM")
            return None

        # Initialize GradCAM with simpler initialization
        cam = GradCAM(
            model=model_for_cam,
            target_layers=[target_layer]
        )
        
        # Generate CAM
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(target_class_idx)]
        )
        
        # Ensure we get the first image if batch
        grayscale_cam = grayscale_cam[0, :]
        
        # Ensure proper normalization of the CAM
        grayscale_cam = np.maximum(grayscale_cam, 0)
        if grayscale_cam.max() != grayscale_cam.min():  # Avoid division by zero
            grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
        
        # Create visualization with more contrast
        visualization = show_cam_on_image(
            original_image,
            grayscale_cam,
            use_rgb=True
        )
        
        # Ensure visualization is in correct format
        if not isinstance(visualization, np.ndarray):
            print("Warning: visualization is not numpy array")
            return None
            
        if visualization.dtype != np.uint8:
            visualization = (visualization * 255).astype(np.uint8)
        
        # Convert to base64
        img = Image.fromarray(visualization)
        buffered = BytesIO()
        img.save(buffered, format="PNG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image file
        if 'image-input' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image-input']
        if not image_file or not image_file.filename:
            return jsonify({'error': 'Empty image file'}), 400

        # Read image bytes once
        image_bytes = image_file.read()
        
        # Preprocess image for model
        image_tensor = preprocess_image(image_bytes)
        image_tensor = image_tensor.to(DEVICE)

        # Process metadata only if model supports it
        meta_tensor = None
        if USE_METADATA and model.module.n_meta_features > 0:
            # Get file size for metadata
            image_file.seek(0, 2)
            file_size = image_file.tell()
            image_file.seek(0)
            
            metadata = {
                'sex': request.form.get('sex', ''),
                'age': request.form.get('age', 0),
                'anatom_site_general': request.form.get('anatom_site_general', ''),
                'n_images': 1.0,
                'image_size': np.log(file_size) if file_size > 0 else 0.0
            }
            meta_tensor = preprocess_metadata(metadata)
            meta_tensor = meta_tensor.to(DEVICE)

        # --- Perform Inference (No Gradients Needed) ---
        try:
            with torch.no_grad():
                # Forward pass
                outputs = model(image_tensor, meta_tensor)
                
                # Temperature scaling
                scaled_outputs = outputs / OPTIMAL_TEMP
                
                # Get probabilities and predictions
                probabilities = torch.softmax(scaled_outputs, dim=1)
                
                # Get top K predictions (K=3)
                k = 3
                top_probs, top_indices = torch.topk(probabilities, k, dim=1)
                
                # Prepare predictions list
                predictions = []
                for i in range(k):
                    class_idx = top_indices[0][i].item()
                    prob = top_probs[0][i].item()
                    class_name = IDX_TO_DIAGNOSIS_DICT[class_idx]
                    predictions.append({
                        'class_name': class_name,
                        'probability': f'{prob:.4f}',
                        'is_melanoma': class_name.lower() == 'melanoma'
                    })

                # Prepare melanoma-specific probability
                melanoma_idx = 6  # Based on provided mapping
                melanoma_prob = probabilities[0][melanoma_idx].item()
                
                # Store top prediction index for Grad-CAM
                top_prediction_idx = top_indices[0][0].item()

        except Exception as pred_err:
            print(f"Prediction error: {str(pred_err)}")
            return jsonify({'error': f'Prediction failed: {str(pred_err)}'}), 500

        # --- Generate Grad-CAM (Gradients Needed Here) ---
        grad_cam_image = None
        if USE_GRADCAM:
            try:
                # Ensure model is still in eval mode
                model.eval() 
                
                target_layer = get_target_layer(model)
                if target_layer is not None:
                    img_for_gradcam = prepare_gradcam_image(image_bytes, IMAGE_SIZE)
                    
                    # We need to potentially re-run the forward pass or ensure the 
                    # input tensor allows grad computation if it was created in no_grad context
                    # Let's re-attach grad if necessary (safer approach)
                    image_tensor_grad = image_tensor.clone().detach().requires_grad_(True)
                    meta_tensor_grad = None
                    if meta_tensor is not None:
                        meta_tensor_grad = meta_tensor.clone().detach().requires_grad_(True)
                        
                    grad_cam_image = generate_gradcam_visualization(
                        model=model, # Pass the main model
                        target_layer=target_layer,
                        input_tensor=image_tensor_grad, # Use tensor that allows grads
                        original_image=img_for_gradcam,
                        target_class_idx=top_prediction_idx, # Use the index from no_grad phase
                        device=DEVICE,
                        meta_tensor=meta_tensor_grad # Use tensor that allows grads
                    )
            except Exception as cam_err:
                print(f"Grad-CAM generation failed: {cam_err}")
                import traceback
                traceback.print_exc() # Print full traceback for CAM error

        # Prepare response
        result = {
            'predictions': predictions,
            'top_prediction': predictions[0],
            'melanoma_probability': f'{melanoma_prob:.4f}',
            'temperature_used': OPTIMAL_TEMP,
            'metadata_used': USE_METADATA and meta_tensor is not None
        }

        # Add original and processed metadata if used
        if USE_METADATA and meta_tensor is not None:
            result['original_metadata'] = metadata # Add the original metadata dict
            # Convert tensor to list for JSON serialization
            processed_metadata_list = meta_tensor.squeeze().tolist() 
            result['processed_metadata'] = {
                'sex_processed': processed_metadata_list[0],
                'age_normalized': f"{processed_metadata_list[1]:.4f}",
                'n_images_log': f"{processed_metadata_list[2]:.4f}",
                'image_size_log': f"{processed_metadata_list[3]:.4f}",
                'location_one_hot': [f"{x:.1f}" for x in processed_metadata_list[4:]] # Format one-hot vector
            }
            # Optionally add location column names for clarity
            result['processed_metadata']['location_columns'] = LOCATION_COLS_LIST

        if grad_cam_image:
            result['grad_cam_image'] = grad_cam_image

        return jsonify(result)

    except Exception as e:
        print(f"General error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("Starting Flask application...")
    app.run(debug=True, port=5000) 