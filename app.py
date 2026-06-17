import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)

st.set_page_config(page_title="Object Identifier", page_icon="🔍")


@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")


model = load_model()

st.title("🔍 Object Identifier")
st.write("Upload a photo and I'll guess what's in it.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Your image", width="stretch")

    with st.spinner("Identifying..."):
        resized = image.resize((224, 224))
        batch = preprocess_input(np.expand_dims(np.array(resized), axis=0))
        preds = decode_predictions(model.predict(batch), top=3)[0]

    st.subheader("Top guesses")
    for _, label, score in preds:
        name = label.replace("_", " ").title()
        st.write(f"{name} — {score * 100:.1f}%")
        st.progress(float(score))
