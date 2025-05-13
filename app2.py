import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import requests
from io import BytesIO

# Load Model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Caption Generation Function
def generate_caption(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption

# Streamlit App UI
st.set_page_config(page_title="Image Captioning", layout="centered")
st.title(" Image Captioning")
st.write("Upload an image or provide an image URL to generate a caption.")

# File Upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# URL Input
image_url = st.text_input("Or enter an image URL:")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
elif image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption="Image from URL", use_column_width=True)
    except:
        st.error("Invalid URL or unable to fetch image.")

# Generate Caption
if image is not None:
    if st.button("Generate Caption ✨"):
        with st.spinner("Generating caption..."):
            caption = generate_caption(image)
        st.success("Caption: " + caption)
