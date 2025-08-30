import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np
import io

# Load Stable Diffusion (first run may take time & need GPU for good results)
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")  # change to "cuda" if GPU available
    return pipe

pipe = load_model()

st.title("🖼️ Text to Image + Image Transformations")

tab1, tab2 = st.tabs(["Text ➝ Image", "Image ➝ Transform"])

# --- TEXT TO IMAGE ---
with tab1:
    st.subheader("Generate Image from Text")
    prompt = st.text_input("Enter your prompt:", "A scenic view of mountains during sunset in watercolor style")
    if st.button("Generate Image"):
        with st.spinner("Generating..."):
            image = pipe(prompt).images[0]
            st.image(image, caption="Generated Image", use_column_width=True)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            st.download_button("Download Image", buf.getvalue(), "generated.png", "image/png")

# --- IMAGE TRANSFORMATIONS ---
with tab2:
    st.subheader("Upload Image for Transformation")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=True)

        option = st.selectbox("Choose Transformation", 
                              ["Grayscale", "Sepia", "Invert", "Cartoon Effect", "Enhance Colors"])

        if st.button("Apply Transformation"):
            img_np = np.array(image)

            if option == "Grayscale":
                result = ImageOps.grayscale(image)

            elif option == "Sepia":
                sepia = np.array(image)
                tr = [0.393, 0.769, 0.189]
                tg = [0.349, 0.686, 0.168]
                tb = [0.272, 0.534, 0.131]
                sepia = np.clip(sepia @ [[tr[0], tg[0], tb[0]],
                                         [tr[1], tg[1], tb[1]],
                                         [tr[2], tg[2], tb[2]]], 0, 255)
                result = Image.fromarray(sepia.astype(np.uint8))

            elif option == "Invert":
                result = ImageOps.invert(image)

            elif option == "Cartoon Effect":
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                gray = cv2.medianBlur(gray, 5)
                edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY, 9, 9)
                color = cv2.bilateralFilter(img_np, 9, 250, 250)
                cartoon = cv2.bitwise_and(color, color, mask=edges)
                result = Image.fromarray(cartoon)

            elif option == "Enhance Colors":
                enhancer = ImageEnhance.Color(image)
                result = enhancer.enhance(1.8)

            st.image(result, caption=f"Transformed: {option}", use_column_width=True)

            buf = io.BytesIO()
            result.save(buf, format="PNG")
            st.download_button("Download Transformed Image", buf.getvalue(), f"{option.lower()}.png", "image/png")
