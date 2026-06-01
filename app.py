import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model.model import model, get_device

device = get_device()

classes = ["female", "male"]

model.load_state_dict(
    torch.load("resnet_gender_model.pth", map_location=device)
)

model.to(device)
model.eval()

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

st.title("Male|Female Classification")

uploaded_file = st.file_uploader(
    "Upload file",
    type=['jpg', 'png', 'jpeg']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded image")

    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    st.write(f"Prediction: {classes[pred]}")
    st.write(f"Confidence: {probs[0][pred].item():.4f}")




