import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.nn import functional as F
from PIL import Image
import io
import sys

MODEL_PATH = "best_hybrid_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

DEFECT_CLASS_NAMES = ['Vertical', 'hole', 'horizontal', 'lines', 'stain']


class HierarchicalFabricModel(nn.Module):
    def __init__(self, num_defect_classes=5, freeze_early_backbone=True):
        super(HierarchicalFabricModel, self).__init__()

        self.backbone = models.resnet50(weights=None)

        for param in self.backbone.parameters():
            param.requires_grad = True

        if freeze_early_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith('layer4'):
                    param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fabric_head = nn.Linear(256, 2)
        self.defect_head = nn.Linear(256, 2)
        self.type_head = nn.Linear(256, num_defect_classes)

    def forward(self, x):
        features = self.backbone(x)
        x_proj = self.shared_fc(features)
        fabric_logits = self.fabric_head(x_proj)
        defect_logits = self.defect_head(x_proj)
        type_logits = self.type_head(x_proj)
        return fabric_logits, defect_logits, type_logits


def test_model():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = HierarchicalFabricModel(num_defect_classes=5, freeze_early_backbone=True)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully.")
        print(f"  Architecture: HierarchicalFabricModel (ResNet50 + 3 Heads)")
        print(f"  Heads: Fabric (2) | Defect (2) | Type (5)")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n--- Running Inference Tests ---")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def run_test(name, tensor):
        with torch.no_grad():
            fabric_logits, defect_logits, type_logits = model(tensor)
            fabric_probs = F.softmax(fabric_logits, dim=1).squeeze().cpu().numpy()
            defect_probs = F.softmax(defect_logits, dim=1).squeeze().cpu().numpy()
            type_probs = F.softmax(type_logits, dim=1).squeeze().cpu().numpy()

        is_fabric = fabric_probs.argmax()
        is_defective = defect_probs.argmax()
        defect_type = DEFECT_CLASS_NAMES[type_probs.argmax()]

        print(f"\nInput: {name}")
        print(f"  Stage 1 - Fabric: {'Yes' if is_fabric else 'No'} "
              f"(conf: {fabric_probs[is_fabric]:.4f})")

        if is_fabric:
            print(f"  Stage 2 - Defect: {'Yes' if is_defective else 'No'} "
                  f"(conf: {defect_probs[is_defective]:.4f})")
            if is_defective:
                print(f"  Stage 3 - Type: {defect_type} "
                      f"(conf: {type_probs[type_probs.argmax()]:.4f})")
            else:
                print(f"  → Classification: defect free")
        else:
            print(f"  → Classification: non fabric")

    # Random noise
    noise = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    run_test("Random Noise", noise)

    # All black image (normalized)
    black = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    black_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(black.squeeze()).unsqueeze(0).to(DEVICE)
    run_test("Black Image", black_norm)

    # All white image (normalized)
    white = torch.ones(1, 3, IMG_SIZE, IMG_SIZE)
    white_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(white.squeeze()).unsqueeze(0).to(DEVICE)
    run_test("White Image", white_norm)


if __name__ == "__main__":
    test_model()
