import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image, ImageOps
from pathlib import Path
import plotly.express as px

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Handwritten Digit Classification", page_icon="✍️", layout="wide")

@st.cache_resource
def load_assets():
    svm = joblib.load(ROOT / "models" / "svm_digit_model.joblib")
    rf = joblib.load(ROOT / "models" / "random_forest_digit_model.joblib")
    metrics = pd.read_csv(ROOT / "reports" / "model_metrics.csv")
    return svm, rf, metrics

svm_model, rf_model, metrics = load_assets()

def prepare_image(image):
    img = ImageOps.exif_transpose(image).convert("L")
    # Typical uploaded page images have a light background and dark ink.
    arr = np.asarray(img, dtype=np.float32)
    if arr.mean() > 127:
        arr = 255.0 - arr
    # Crop around visible ink to reduce surrounding blank space.
    mask = arr > 25
    if mask.any():
        ys, xs = np.where(mask)
        arr = arr[max(0, ys.min()-2):min(arr.shape[0], ys.max()+3),
                  max(0, xs.min()-2):min(arr.shape[1], xs.max()+3)]
    digit = Image.fromarray(np.uint8(np.clip(arr, 0, 255))).resize((8, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(digit, dtype=np.float32) / 255.0 * 16.0
    return pixels, digit

st.title("✍️ Handwritten Digit Classification")
st.caption("Machine Learning project | Python • Streamlit • Scikit-learn")
st.write("Upload an image of a single handwritten digit (0–9). The application predicts the digit using trained machine-learning models.")

tab1, tab2, tab3 = st.tabs(["🔍 Predict a Digit", "📊 Model Performance", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg", "bmp"])
        model_choice = st.selectbox("Choose classification model", ["Support Vector Machine (SVM)", "Random Forest"])
        st.info("Tip: Upload one large, clearly written digit with good contrast. This demo uses the 8×8 Scikit-learn Digits dataset.")
    with col2:
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded image", width=240)
            pixels, preview = prepare_image(image)
            st.image(preview.resize((240, 240)), caption="Preprocessed 8×8 input", width=240)
            vector = pixels.reshape(1, -1)
            model = svm_model if model_choice.startswith("Support") else rf_model
            prediction = int(model.predict(vector)[0])
            st.success(f"Predicted digit: **{prediction}**")
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(vector)[0]
                prob_df = pd.DataFrame({"Digit": list(range(10)), "Probability": probabilities})
                fig = px.bar(prob_df, x="Digit", y="Probability", title="Prediction probabilities")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Upload an image to see a prediction.")
            st.image(str(ROOT / "sample_images" / "sample_digit_7.png"), caption="Example input image", width=150)

with tab2:
    st.subheader("Held-out test-set performance")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    fig = px.bar(metrics, x="Model", y="Accuracy (%)", text_auto=".2f", title="Model accuracy comparison", range_y=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Scores are calculated on a stratified test split of the built-in Scikit-learn Digits dataset. They are not manually entered.")

with tab3:
    st.subheader("Project overview")
    st.write("""
    This project classifies handwritten digits from 0 to 9. It uses the built-in
    Scikit-learn Digits dataset, which contains 1,797 grayscale images represented
    as 8×8 pixel arrays.

    **Workflow:** image input → grayscale conversion → contrast normalization →
    resize to 8×8 → trained classifier → predicted digit.

    **Algorithms:** Support Vector Machine (SVM) and Random Forest.

    **Limitations:** This educational demo is trained on small 8×8 digit images.
    Photographs of handwritten digits may need cropping, good lighting, and clear
    contrast. It is not a full-resolution MNIST production model.
    """)
    st.markdown("**Technology:** Python, Streamlit, Scikit-learn, Pandas, Plotly, Pillow and Joblib.")
