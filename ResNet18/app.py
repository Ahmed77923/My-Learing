from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


st.set_page_config(
    page_title="CIFAR Vision Lab",
    page_icon="CV",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "artifacts" / "best_resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #101820;
        --paper: #f2f5f3;
        --cyan: #20d8d0;
        --amber: #f2b84b;
        --muted: #68737a;
    }

    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: var(--ink); }
    [data-testid="stSidebar"] * { color: #eaf2ef; }
    .block-container { max-width: 1180px; padding-top: 3.5rem; }
    h1, h2, h3, p, label { font-family: 'Space Grotesk', sans-serif; }
    h1 { font-size: clamp(2.8rem, 6vw, 5.7rem); line-height: .92; letter-spacing: 0; max-width: 760px; }
    h2 { font-size: 1.45rem; }
    .eyebrow { color: #168c88; font-family: 'DM Mono', monospace; font-size: .72rem; letter-spacing: .08em; margin-bottom: .8rem; }
    .lede { color: var(--muted); font-size: 1.08rem; max-width: 590px; line-height: 1.55; }
    .rule { border-top: 1px solid #cbd5d1; margin: 2.2rem 0; }
    .metric-label { color: #8e9a9c; font-family: 'DM Mono', monospace; font-size: .68rem; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value { color: var(--cyan); font-family: 'Space Grotesk', sans-serif; font-size: 1.55rem; font-weight: 700; margin-top: .25rem; }
    .result-box { background: var(--ink); border-left: 5px solid var(--cyan); padding: 1.35rem 1.5rem; margin: 1rem 0 1.4rem; }
    .result-box .metric-label { color: #a9bab7; }
    .result-box .metric-value { color: #fff; font-size: 2.2rem; }
    .stButton > button { background: var(--ink); color: white; border: 0; border-radius: 0; font-family: 'Space Grotesk', sans-serif; font-weight: 600; padding: .7rem 1.3rem; }
    .stButton > button:hover { background: #243640; color: var(--cyan); border: 0; }
    [data-testid="stFileUploader"] { border: 1px dashed #9caeaa; padding: .4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    image_size = checkpoint.get("image_size", 224)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, class_names, image_size, checkpoint


def predict(image, model, class_names, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probabilities = F.softmax(model(tensor), dim=1)[0]
    scores, indices = torch.topk(probabilities, k=min(5, len(class_names)))
    return [(class_names[index.item()], score.item()) for score, index in zip(scores, indices)]


st.markdown('<div class="eyebrow">RESNET18 / CIFAR-10 / FEATURE EXTRACTION</div>', unsafe_allow_html=True)
st.title("See what the model sees.")
st.markdown(
    '<p class="lede">Drop in an image and inspect the ResNet18 prediction. The model reads it at 224 pixels, then ranks the five most likely visual classes.</p>',
    unsafe_allow_html=True,
)

try:
    model, class_names, image_size, checkpoint = load_model()
except FileNotFoundError:
    st.error(f"Checkpoint not found at {CHECKPOINT_PATH}. Train ResNet18 first.")
    st.stop()
except Exception as error:
    st.error("The checkpoint could not be loaded.")
    st.code(str(error))
    st.stop()


with st.sidebar:
    st.markdown("## Model notes")
    st.caption("A frozen ImageNet backbone with a CIFAR-10 classifier head.")
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-label">Runtime</div><div class="metric-value">{"GPU" if DEVICE.type == "cuda" else "CPU"}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-label" style="margin-top:1.2rem">Input size</div><div class="metric-value">{image_size} px</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-label" style="margin-top:1.2rem">Best epoch</div><div class="metric-value">{checkpoint.get("best_epoch", "-")}</div>', unsafe_allow_html=True)
    st.markdown("<div class=""rule""></div>", unsafe_allow_html=True)
    st.caption("Classes: " + ", ".join(class_names))


st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload an image to classify",
    type=["jpg", "jpeg", "png", "webp"],
    help="Use a clear RGB image for the most useful result.",
)

if uploaded_file is None:
    st.info("Your image will appear here with the prediction breakdown.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    preview, analysis = st.columns([1.05, 1], gap="large")

    with preview:
        st.markdown('<div class="eyebrow">INPUT FRAME</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.caption(f"{image.width} x {image.height} pixels  |  RGB")

    with analysis:
        st.markdown('<div class="eyebrow">CLASSIFICATION</div>', unsafe_allow_html=True)
        if st.button("Classify image", type="primary", use_container_width=True):
            results = predict(image, model, class_names, image_size)
            st.session_state["results"] = results

        results = st.session_state.get("results")
        if results:
            label, confidence = results[0]
            st.markdown(
                f'<div class="result-box"><div class="metric-label">Most likely class</div><div class="metric-value">{label}</div><div style="color:#a9bab7;margin-top:.35rem">{confidence:.1%} confidence</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("#### Ranked possibilities")
            for rank, (name, score) in enumerate(results, start=1):
                st.markdown(f"**{rank:02d}  {name}**  ·  {score:.1%}")
                st.progress(score)
        else:
            st.caption("Run classification when you are ready.")

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
st.caption("Predictions are probabilities produced by the trained classifier, not guarantees of visual identity.")