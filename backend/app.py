import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.nn import functional as F
from PIL import Image
import io
import base64
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for frontend app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = "best_hybrid_model.pth"  # The GAN-augmented hierarchical model
IMG_SIZE = 224  # Must match the training image size from the notebook
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defect class names — matching the notebook's defect_mapping:
#   0: Vertical, 1: hole, 2: horizontal, 3: lines, 4: stain
DEFECT_CLASS_NAMES = ['Vertical', 'hole', 'horizontal', 'lines', 'stain']
NUM_DEFECT_CLASSES = len(DEFECT_CLASS_NAMES)

# ──────────────────────────────────────────────────────────────────────────────
# Model Definition — Must match the HierarchicalFabricModel used during
# training in the notebook (ResNet50 backbone + 3 hierarchical heads).
# ──────────────────────────────────────────────────────────────────────────────
class HierarchicalFabricModel(nn.Module):
    def __init__(self, num_defect_classes=5, freeze_early_backbone=True):
        super(HierarchicalFabricModel, self).__init__()

        # Load ResNet-50 structure (weights will be loaded from checkpoint)
        self.backbone = models.resnet50(weights=None)

        # Configure fine-tuning parameters (matches training config)
        for param in self.backbone.parameters():
            param.requires_grad = True

        if freeze_early_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith('layer4'):
                    param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Shared Projection Layer
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Hierarchical heads
        self.fabric_head = nn.Linear(256, 2)    # fabric vs non-fabric
        self.defect_head = nn.Linear(256, 2)    # defect vs defect-free
        self.type_head = nn.Linear(256, num_defect_classes)  # defect type

    def forward(self, x):
        features = self.backbone(x)
        x_proj = self.shared_fc(features)
        fabric_logits = self.fabric_head(x_proj)
        defect_logits = self.defect_head(x_proj)
        type_logits = self.type_head(x_proj)
        return fabric_logits, defect_logits, type_logits


model = HierarchicalFabricModel(
    num_defect_classes=NUM_DEFECT_CLASSES,
    freeze_early_backbone=True
)

# Load Weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    print(f"  Architecture : HierarchicalFabricModel (ResNet50 + 3 Heads)")
    print(f"  Heads        : Fabric (2) | Defect (2) | Type ({NUM_DEFECT_CLASSES})")
    print(f"  Defect Types : {DEFECT_CLASS_NAMES}")
    print(f"  Image Size   : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Device       : {DEVICE}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure 'best_hybrid_model.pth' is in the same directory.")

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing — Must match the validation/test transforms from the notebook
# ──────────────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def run_inference(image: Image.Image) -> dict:
    """Hierarchical inference: Fabric → Defect → Defect Type."""
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fabric_logits, defect_logits, type_logits = model(input_tensor)

        fabric_probs = F.softmax(fabric_logits, dim=1).squeeze().cpu().numpy()
        defect_probs = F.softmax(defect_logits, dim=1).squeeze().cpu().numpy()
        type_probs = F.softmax(type_logits, dim=1).squeeze().cpu().numpy()

    # ── DEBUG: Print raw outputs ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[DEBUG] Fabric logits:  {fabric_logits.cpu().numpy()}")
    print(f"[DEBUG] Fabric probs:   [non-fabric={fabric_probs[0]:.4f}, fabric={fabric_probs[1]:.4f}]")
    print(f"[DEBUG] Defect logits:  {defect_logits.cpu().numpy()}")
    print(f"[DEBUG] Defect probs:   [defect-free={defect_probs[0]:.4f}, defective={defect_probs[1]:.4f}]")
    print(f"[DEBUG] Type logits:    {type_logits.cpu().numpy()}")
    print(f"[DEBUG] Type probs:     {dict(zip(DEFECT_CLASS_NAMES, [f'{p:.4f}' for p in type_probs]))}")

    # ── Stage 1: Fabric vs Non-Fabric ────────────────────────────────────
    is_fabric = int(fabric_probs.argmax())  # 0 = non-fabric, 1 = fabric
    fabric_confidence = float(fabric_probs[is_fabric])
    print(f"[DEBUG] → Stage 1: is_fabric={is_fabric} (conf={fabric_confidence:.4f})")

    if not is_fabric:
        # Non-fabric image
        return {
            "prediction": "Non-Fabric",
            "defect_type": "non fabric",
            "confidence": f"{fabric_confidence * 100:.2f}%",
            "confidence_value": round(fabric_confidence, 4),
            "is_defective": False,
            "is_fabric": False,
            "hierarchy": {
                "stage1_fabric": {
                    "result": "Non-Fabric",
                    "confidence": round(float(fabric_probs[0]), 4),
                    "probabilities": {
                        "non_fabric": round(float(fabric_probs[0]), 4),
                        "fabric": round(float(fabric_probs[1]), 4)
                    }
                },
                "stage2_defect": None,
                "stage3_type": None
            },
            "class_probabilities": [
                {"class_name": "non fabric", "probability": round(float(fabric_probs[0]), 4)},
                {"class_name": "fabric", "probability": round(float(fabric_probs[1]), 4)}
            ]
        }

    # ── Stage 2: Defect vs Defect-Free ───────────────────────────────────
    is_defective = int(defect_probs.argmax())  # 0 = defect-free, 1 = defective
    defect_confidence = float(defect_probs[is_defective])

    if not is_defective:
        # Fabric but no defect
        return {
            "prediction": "Normal",
            "defect_type": "defect free",
            "confidence": f"{defect_confidence * 100:.2f}%",
            "confidence_value": round(defect_confidence, 4),
            "is_defective": False,
            "is_fabric": True,
            "hierarchy": {
                "stage1_fabric": {
                    "result": "Fabric",
                    "confidence": round(float(fabric_probs[1]), 4),
                    "probabilities": {
                        "non_fabric": round(float(fabric_probs[0]), 4),
                        "fabric": round(float(fabric_probs[1]), 4)
                    }
                },
                "stage2_defect": {
                    "result": "Defect-Free",
                    "confidence": round(float(defect_probs[0]), 4),
                    "probabilities": {
                        "defect_free": round(float(defect_probs[0]), 4),
                        "defective": round(float(defect_probs[1]), 4)
                    }
                },
                "stage3_type": None
            },
            "class_probabilities": [
                {"class_name": "defect free", "probability": round(float(defect_probs[0]), 4)},
                {"class_name": "defective", "probability": round(float(defect_probs[1]), 4)}
            ]
        }

    # ── Stage 3: Defect Type Classification ──────────────────────────────
    predicted_type_idx = int(type_probs.argmax())
    predicted_type_name = DEFECT_CLASS_NAMES[predicted_type_idx]
    type_confidence = float(type_probs[predicted_type_idx])

    # Build sorted class probabilities for defect types
    class_probabilities = []
    for i, class_name in enumerate(DEFECT_CLASS_NAMES):
        class_probabilities.append({
            "class_name": class_name,
            "probability": round(float(type_probs[i]), 4)
        })
    class_probabilities.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "prediction": "Defect",
        "defect_type": predicted_type_name,
        "confidence": f"{type_confidence * 100:.2f}%",
        "confidence_value": round(type_confidence, 4),
        "is_defective": True,
        "is_fabric": True,
        "hierarchy": {
            "stage1_fabric": {
                "result": "Fabric",
                "confidence": round(float(fabric_probs[1]), 4),
                "probabilities": {
                    "non_fabric": round(float(fabric_probs[0]), 4),
                    "fabric": round(float(fabric_probs[1]), 4)
                }
            },
            "stage2_defect": {
                "result": "Defective",
                "confidence": round(float(defect_probs[1]), 4),
                "probabilities": {
                    "defect_free": round(float(defect_probs[0]), 4),
                    "defective": round(float(defect_probs[1]), 4)
                }
            },
            "stage3_type": {
                "result": predicted_type_name,
                "confidence": round(type_confidence, 4),
                "probabilities": {
                    name: round(float(type_probs[i]), 4)
                    for i, name in enumerate(DEFECT_CLASS_NAMES)
                }
            }
        },
        "class_probabilities": class_probabilities
    }


@app.get("/")
def home():
    return {"message": "Fabric Defect Detection API is running (Hierarchical Model)"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        return run_inference(image)
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# Live Camera Frame Endpoint — accepts base64-encoded JPEG/PNG frames
# ──────────────────────────────────────────────────────────────────────────────
class FramePayload(BaseModel):
    frame: str  # base64-encoded image data (data URI or raw base64)

@app.post("/predict/frame")
async def predict_frame(payload: FramePayload):
    try:
        # Strip optional data-URI prefix (e.g. "data:image/jpeg;base64,")
        frame_data = payload.frame
        if "," in frame_data:
            frame_data = frame_data.split(",", 1)[1]

        image_bytes = base64.b64decode(frame_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return run_inference(image)
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
