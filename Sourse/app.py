
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# -------------------------
# Configuration
# -------------------------
st.set_page_config(
    page_title="SimpleCNN Image Classifier",
    page_icon="🧠",
    layout="wide",
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "best_simple_cnn.pth"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# -------------------------
# Model architecture
# Must match the training model exactly
# -------------------------
# Input → Conv2d → ReLU → MaxPool2d → Conv2d → ReLU → MaxPool2d → Flatten → Linear
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(   # layer 1
            nn.Conv2d(in_channels= 3, out_channels=16, kernel_size=3, stride=1 , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1 , padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 8 * 8 , out_features=10)
        )
        
        
    def forward(self, x):
        x = self.feature(x)
        return  self.classifier(x)
    
    



# -------------------------
# Load model once
# -------------------------
@st.cache_resource
def load_model():
    model = SimpleCNN().to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


# -------------------------
# Image preprocessing
# Must match training normalization
# -------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),     # i added this to augment the data
    transforms.RandomCrop(32, padding=4),  # i added this to augment the data
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])


# -------------------------
# User interface
# -------------------------
st.title("🧠 SimpleCNN Image Classifier")
st.write(
    "Upload an image to see what your trained "
    "CIFAR-10 model predicts."
)

st.caption(f"Running on: {DEVICE}")

try:
    model = load_model()
    st.success("Model loaded successfully.")
except FileNotFoundError:
    st.error(
        f"Model file not found: {MODEL_PATH}. "
        "Place best_simple_cnn.pth next to app.py."
    )
    st.stop()
except Exception as error:
    st.error(
        "Could not load the checkpoint. Ensure the "
        "SimpleCNN architecture and layer names match "
        "the training code exactly."
    )
    st.code(str(error))
    st.stop()


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("Input image")
        st.image(image, use_container_width=True)
        st.caption(f"Original size: {image.size[0]} × {image.size[1]}")

    if st.button("Classify image", type="primary"):
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = F.softmax(logits, dim=1)[0]

        top_probs, top_indices = torch.topk(
            probabilities, k=5
        )

        predicted_index = top_indices[0].item()
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = top_probs[0].item() * 100

        with right:
            st.subheader("Prediction")
            st.metric("Predicted class", predicted_class)
            st.metric("Confidence", f"{confidence:.2f}%")

            st.subheader("Top 5 predictions")

            for prob, idx in zip(top_probs, top_indices):
                name = CLASS_NAMES[idx.item()]
                score = prob.item()

                st.write(f"**{name}** — {score:.2%}")
                st.progress(score)

        st.info(
            "Confidence is the model's predicted probability, "
            "not a guarantee that the prediction is correct."
        )

st.divider()
st.caption(
    "Model: SimpleCNN | Dataset: CIFAR-10 | "
    "Input size: 32 × 32 RGB"
)
