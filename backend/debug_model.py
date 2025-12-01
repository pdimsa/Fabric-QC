import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import sys

MODEL_PATH = "best_hybrid_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128

def get_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

def test_model():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = get_model()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create dummy inputs
    print("\n--- Running Inference Tests ---")
    
    # Random Noise
    noise = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    with torch.no_grad():
        output = model(noise)
        prob = torch.sigmoid(output).item()
    print(f"Input: Random Noise | Output Logit: {output.item():.4f} | Probability: {prob:.4f} | Prediction: {'Defect' if prob > 0.5 else 'Normal'}")

    # All Black Image (0s)
    black = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    # Normalize expected input range [0, 1] -> [-1, 1]
    # ToTensor scales [0, 255] to [0, 1]. Normalize (0.5, 0.5) makes 0 -> -1.
    black_norm = (black - 0.5) / 0.5
    with torch.no_grad():
        output = model(black_norm)
        prob = torch.sigmoid(output).item()
    print(f"Input: Black Image  | Output Logit: {output.item():.4f} | Probability: {prob:.4f} | Prediction: {'Defect' if prob > 0.5 else 'Normal'}")

    # All White Image (1s)
    white = torch.ones(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    # Normalize: 1 -> (1 - 0.5)/0.5 = 1.
    white_norm = (white - 0.5) / 0.5
    with torch.no_grad():
        output = model(white_norm)
        prob = torch.sigmoid(output).item()
    print(f"Input: White Image  | Output Logit: {output.item():.4f} | Probability: {prob:.4f} | Prediction: {'Defect' if prob > 0.5 else 'Normal'}")

if __name__ == "__main__":
    test_model()
