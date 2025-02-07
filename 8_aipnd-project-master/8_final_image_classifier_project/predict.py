import torch
from torchvision import models
from torchvision.models import VGG16_Weights, VGG13_Weights
from PIL import Image
import numpy as np
import argparse
import json
import os

def get_input_args():
    """
    Parses command line arguments for prediction of image class.
    """
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained model.")

    # Positional arguments
    parser.add_argument('image_path', type=str, help='Path of the image to be processed')
    parser.add_argument('checkpoint', type=str, help='Path of the saved model checkpoint')

    # Optional arguments
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='', help='Path to JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    return parser.parse_args()

def load_checkpoint(filepath):
    """
    Loads a checkpoint and rebuilds the model.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint file '{filepath}' not found.")

    checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)  # ✅ Use weights_only=True

    arch = checkpoint['arch']
    
    # Handle different architectures with correct weights
    available_models = {
        "vgg16": (models.vgg16, VGG16_Weights.IMAGENET1K_V1),
        "vgg13": (models.vgg13, VGG13_Weights.IMAGENET1K_V1)
    }

    if arch not in available_models:
        raise ValueError(f"Unsupported architecture '{arch}'. Supported: {list(available_models.keys())}")

    model_fn, weights_enum = available_models[arch]
    model = model_fn(weights=weights_enum)  # ✅ Updated for PyTorch 0.13+
    
    # Freeze parameters
    for param in model.features.parameters():
        param.requires_grad = False

    # Rebuild classifier
    hidden_units = checkpoint['hidden_units']
    input_features = model.classifier[0].in_features
    from torch import nn
    classifier = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a FloatTensor
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    img = Image.open(image_path)

    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))

    # Center crop
    left = (img.width - 224) / 2
    top = (img.height - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))

    # Convert to numpy
    np_image = np.array(img) / 255
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    # Reorder dimensions
    np_image = np_image.transpose((2, 0, 1))

    return torch.from_numpy(np_image).type(torch.FloatTensor)

def predict(image_path, model, topk=5, device='cpu'):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """
    model.to(device)
    model.eval()

    # Process image
    img_tensor = process_image(image_path)
    img_tensor = img_tensor.unsqueeze_(0).to(device)

    with torch.no_grad():
        output = model.forward(img_tensor)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)

    # Convert indices to classes
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[c.item()] for c in top_class[0]]

    return top_p[0].tolist(), top_classes

def main():
    args = get_input_args()

    # 1. Load checkpoint
    model = load_checkpoint(args.checkpoint)

    # 2. Select device
    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")

    # 3. Predict
    probs, classes = predict(args.image_path, model, topk=args.top_k, device=device)

    # 4. If category_names is provided, map class -> flower name
    if args.category_names and os.path.isfile(args.category_names):
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name.get(cls, "Unknown") for cls in classes]
    else:
        class_names = classes

    # 5. Print results
    print("\nTop K Classes and Probabilities:")
    for c, p in zip(class_names, probs):
        print(f"{c}: {p:.3f}")

if __name__ == '__main__':
    main()
