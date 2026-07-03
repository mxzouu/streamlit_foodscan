import streamlit as st
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# ──────────────────────────────────────────────────────────────
#  FOODSCAN — CALORIE ESTIMATOR
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FoodScan — Calorie Estimator",
    page_icon="🍽️",
    layout="centered"
)

# ── PAGE HEADER ───────────────────────────────────────────────
st.title("🍽️ FoodScan")
st.subheader("Food Calorie Estimator")
st.write(
    "Upload a photo of a food dish and the model will estimate "
    "its total calorie content."
)
st.divider()

# ── CONFIG — doit matcher EXACTEMENT le notebook d'entraînement ──
# Reporte ici la valeur imprimée par "BACKBONE choisi:" sur Kaggle
BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"
IMG_SIZE = 320
WEIGHTS_PATH = "best_model.pt"

# Poids hébergés sur Google Drive (fichier > 100 Mo, limite GitHub)
GDRIVE_FILE_ID = "1cEjDbpMl9ZWhZ7mbps9216hFm7klx-nd"


def ensure_weights_downloaded():
    """Télécharge les poids depuis Google Drive si absents en local."""
    if not os.path.exists(WEIGHTS_PATH):
        import gdown
        with st.spinner("Downloading model weights (first launch only)..."):
            gdown.download(id=GDRIVE_FILE_ID, output=WEIGHTS_PATH, quiet=False)


# ── MODEL DEFINITION ─────────────────────────────────────────
# Identique à FoodModel dans le notebook Kaggle
class CalorieEstimator(nn.Module):
    def __init__(self, backbone=BACKBONE):
        super().__init__()
        # pretrained=False : pas besoin de retélécharger les poids ImageNet,
        # on va charger nos propres poids fine-tunés juste après
        self.bb = timm.create_model(backbone, pretrained=False, num_classes=0)
        feat = self.bb.num_features
        self.head = nn.Linear(feat + 1, 1)

    def forward(self, img, dom):
        f = self.bb(img)
        return self.head(torch.cat([f, dom], dim=1)).squeeze(1)


# ── MODEL LOADING ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    ensure_weights_downloaded()
    model = CalorieEstimator()
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ── PREPROCESSING ─────────────────────────────────────────────
# Reproduit A.Resize + A.Normalize() (défauts ImageNet) + ToTensorV2()
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_domain_flag(filename: str) -> torch.Tensor:
    """Reproduit le proxy de domaine utilisé à l'entraînement (ext='jpg' -> 1.0)."""
    ext = filename.split(".")[-1].lower()
    val = 1.0 if ext in ("jpg", "jpeg") else 0.0
    return torch.tensor([[val]], dtype=torch.float32)


# ── INFERENCE ────────────────────────────────────────────────
def predict(image: Image.Image, filename: str, model: nn.Module) -> float:
    transform = get_transform()
    tensor = transform(image).unsqueeze(0)
    dom = get_domain_flag(filename)

    with torch.no_grad():
        log_pred = model(tensor, dom)

    # le modèle prédit log(1 + calories) -> on repasse en kcal
    predicted_calories = float(np.expm1(log_pred.item()))
    predicted_calories = max(0.0, predicted_calories)  # garde-fou

    return round(predicted_calories, 1)


# ── MAIN UI ──────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo of a single food dish"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        st.write("")
        st.write("")
        with st.spinner("Estimating calories..."):
            try:
                model      = load_model()
                prediction = predict(image, uploaded_file.name, model)

                st.metric(
                    label="Estimated Calories",
                    value=f"{prediction:.0f} kcal"
                )

            except FileNotFoundError:
                st.error(
                    f"Model weights not found ('{WEIGHTS_PATH}'). "
                    "Make sure best_model.pt is in the same folder as app.py."
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.divider()
st.caption(
    "FoodScan Challenge — Deep Learning For Images | "
    "M2 IASD Apprenticeship | Université Paris Dauphine - PSL"
)