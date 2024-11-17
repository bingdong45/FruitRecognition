import streamlit as st
import torch
import numpy as np
from model import GoogleNet
from torchvision import transforms
import io
from PIL import Image

st.title("_What's that_ :green[Fruit] :apple:")
st.header(":arrow_up_small: Mode")
st.markdown("<h4>Directions</h4>", unsafe_allow_html=True)
st.markdown("""
1. Click on upload box  
2. Select a picture of fruit you want to identify  
3. Then scroll down  
""")

st.markdown("### Upload your image below:")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        preprocess = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.ToTensor()
        ])

        input_tensor = preprocess(image).unsqueeze(0)

        model = GoogleNet()
        weights_path = '../weights/model_weight.pth'
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()

        with torch.inference_mode():
            output = model(input_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
            labels = ['Apple', 'Avocado', 'Banana', 'Cherry', 'Kiwi', 
                      'Mango', 'Orange', 'Pineapple', 'Strawberries', 'Watermelon']
            predicted_fruit = labels[predicted_idx]

            st.markdown(f"<h3>Predicted Fruit: {predicted_fruit}</h3>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
