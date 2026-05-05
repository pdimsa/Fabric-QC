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
MODEL_PATH = "best_hybrid_model.pth" # Ensure this file is present or path is correct
IMG_SIZE = 224  # Must match the training image size from the notebook
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names — alphabetically ordered as produced by sklearn's LabelEncoder
# matching the notebook's label encoding:
#   0: Vertical, 1: defect free, 2: hole, 3: horizontal, 4: lines, 5: stain
CLASS_NAMES = ['Vertical', 'defect free', 'hole', 'horizontal', 'lines', 'stain']
NUM_CLASSES = len(CLASS_NAMES)

# ──────────────────────────────────────────────────────────────────────────────
# Model Definition — Must match the ResNet50Classifier used during training
# in the gan-fabric-model.ipynb notebook.
# ──────────────────────────────────────────────────────────────────────────────
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super(ResNet50Classifier, self).__init__()

        # Load ResNet50 structure (weights will be loaded from checkpoint)
        self.backbone = models.resnet50(pretrained=False)

        # Freeze backbone if specified (matches training config)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final classification layer with custom head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


model = ResNet50Classifier(num_classes=NUM_CLASSES, freeze_backbone=True)

# Load Weights
try:
    # map_location ensures it loads on CPU if CUDA not available
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    print(f"  Architecture : ResNet50 + Custom Head ({NUM_CLASSES} classes)")
    print(f"  Classes      : {CLASS_NAMES}")
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
    """Shared inference logic for both upload and live-cam endpoints."""
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1).squeeze().cpu().numpy()

    predicted_idx = int(probabilities.argmax())
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    is_defective = predicted_class != 'defect free'

    class_probabilities = []
    for i, class_name in enumerate(CLASS_NAMES):
        class_probabilities.append({
            "class_name": class_name,
            "probability": round(float(probabilities[i]), 4)
        })
    class_probabilities.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "prediction": "Defect" if is_defective else "Normal",
        "defect_type": predicted_class,
        "confidence": f"{confidence * 100:.2f}%",
        "confidence_value": round(confidence, 4),
        "is_defective": is_defective,
        "class_probabilities": class_probabilities
    }


@app.get("/")
def home():
    return {"message": "Fabric Defect Detection API is running"}


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
