import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse
import json
import os

def get_input_args():
    """
    Parses command line arguments for training a new network.
    """
    parser = argparse.ArgumentParser(description="Train a new neural network on a dataset of images.")

    # Positional argument: data directory
    parser.add_argument('data_dir', type=str, help='Directory containing the training/valid/test image subfolders')

    # Optional: set directory to save checkpoints
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the trained model checkpoint')

    # Optional: choose architecture
    parser.add_argument('--arch', type=str, default='vgg16',
                        help="Choose architecture from torchvision.models (e.g., 'vgg16', 'resnet50')")

    # Optional: set hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units in the feedforward classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')

    # Optional: use GPU
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    return parser.parse_args()

def load_data(data_dir):
    """
    Loads the image data with appropriate transforms and returns DataLoaders and image datasets.
    """
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir  = os.path.join(data_dir, 'test')

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=test_transforms)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=64, shuffle=False)

    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader

def build_model(arch='vgg16', hidden_units=512):
    """
    Builds a pretrained model, sets up the classifier, and returns the model.
    """
    available_models = {
        "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
        "vgg13": (models.vgg13, models.VGG13_Weights.IMAGENET1K_V1),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1)
    }

    if arch not in available_models:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose from {list(available_models.keys())}")

    # Load Pretrained Model
    model_fn, weights_enum = available_models[arch]
    model = model_fn(weights=weights_enum)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define classifier
    if "vgg" in arch:
        input_features = model.classifier[0].in_features
        classifier = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier
    elif "resnet" in arch:
        input_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )

    return model

def validate_model(model, loader, criterion, device):
    """
    Validates the model on the validation loader and returns the loss and accuracy.
    """
    model.eval()
    loss = 0
    accuracy = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            batch_loss = criterion(output, labels)
            loss += batch_loss.item()

            # Accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = (top_class == labels.view(*top_class.shape))
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return loss / len(loader), accuracy / len(loader)

def train_model(model, train_loader, valid_loader, device, epochs=5, lr=0.001):
    """
    Trains the model using the given loaders and returns the trained model.
    """
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print_every = 40
    steps = 0

    for e in range(epochs):
        running_loss = 0
        model.train()

        for images, labels in train_loader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                val_loss, val_accuracy = validate_model(model, valid_loader, criterion, device)
                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {val_loss:.3f}.. "
                      f"Validation accuracy: {val_accuracy:.3f}")
                running_loss = 0
                model.train()

    return model

def save_checkpoint(model, train_dataset, save_dir='.', arch='vgg16', hidden_units=512):
    """
    Saves the trained model as a checkpoint.
    """
    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }

    save_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")

def main():
    args = get_input_args()
    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")

    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = load_data(args.data_dir)
    model = build_model(arch=args.arch, hidden_units=args.hidden_units)
    model.to(device)

    trained_model = train_model(model, train_loader, valid_loader, device, epochs=args.epochs, lr=args.learning_rate)
    save_checkpoint(trained_model, train_dataset, save_dir=args.save_dir, arch=args.arch, hidden_units=args.hidden_units)

if __name__ == '__main__':
    main()
