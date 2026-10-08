"""
ECG Normal/Abnormal classifier demo.

Personal learning exercise (separate from ChronicAI) — a ResNet18 fine-tuned
on 1,682 photographed 12-lead ECGs (Khan & Hussain, Mendeley Data, CC BY 4.0)
to classify Normal vs Abnormal. Upload any ECG image to test it live.

The model only ever outputs Normal/Abnormal — it does not attempt to name
the specific diagnosis (WPW, RBBB, etc.), and this is a personal learning
project, not a validated clinical tool.
"""
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "ecg_classifier.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def predict(model, image: Image.Image):
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        prob_abnormal = torch.sigmoid(model(tensor).squeeze()).item()
    return prob_abnormal


st.set_page_config(page_title="ECG Normal/Abnormal Classifier", layout="wide")

# ---- Sidebar: model details, tucked behind a button ----
with st.sidebar:
    st.header("ECG Classifier")
    show_details = st.button("About this model")
    if show_details:
        st.session_state["show_details"] = not st.session_state.get("show_details", False)
    if st.session_state.get("show_details", False):
        st.markdown(
            """
            **Recipe:** ResNet18, pretrained on ImageNet, fine-tuned in two
            stages (frozen-backbone feature extraction, then partial
            fine-tuning of `layer3`/`layer4` with checkpointing on best
            validation accuracy).

            **Training data:** 1,682 photographed 12-lead ECGs — 859 Normal,
            823 Abnormal (merged from Abnormal Heartbeat / MI / Previous MI)
            — from Khan & Hussain, *"ECG Images dataset of Cardiac and
            COVID-19 Patients"*, Mendeley Data, CC BY 4.0.

            **Validation result:** 85.5% accuracy, sensitivity and
            specificity both at 85.5%, on a held-out 20% split.
            """
        )

# ---- Main area ----
st.markdown(
    "<h1 style='color:#2563eb; font-size:2.4rem;'>Upload an ECG to Test</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Upload a 12-lead ECG image (photo or scan). The model classifies it as "
    "Normal or Abnormal only — it does not name a specific diagnosis."
)

uploaded_file = st.file_uploader("ECG image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded ECG", use_column_width=True)

    if st.button("Run model", type="primary"):
        model = load_model()
        prob_abnormal = predict(model, image)
        model_call = "Abnormal" if prob_abnormal >= 0.5 else "Normal"

        st.markdown("---")
        st.markdown(f"**Model's raw probability of Abnormal:** {prob_abnormal:.3f}")
        if model_call == "Abnormal":
            st.error(f"Model says: **{model_call}**")
        else:
            st.success(f"Model says: **{model_call}**")

        st.caption(
            "This model does binary screening (Normal vs Abnormal) only — it is a "
            "personal learning project, not a validated clinical tool, and should "
            "not be used for real diagnostic decisions."
        )
