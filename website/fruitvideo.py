import streamlit as st
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import io
import numpy as np
from model import GoogleNet

st.title("_What's that_ :green[Fruit] :apple:")
st.header(":camera: Mode")

st.markdown("""
**Directions:**  
1. Click on the camera icon to use your camera  
2. Aim your camera at the fruit you want to identify and click “Take a Photo”  
3. Click “Search” to see your results!  
""")

photo = st.camera_input("Take a picture here:")

if photo is not None:
    try:
        image = Image.open(io.BytesIO(photo.read())).convert('RGB')
        st.image(image, caption="Captured Image", use_column_width=True)

        preprocess = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.ToTensor()
        ])

        input_tensor = preprocess(image).unsqueeze(0)

        model = GoogleNet()
        weights_path = '../weights/model_weight.pth'
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()

        # Make prediction
        with torch.inference_mode():
            predictions = model(input_tensor)
            predicted_idx = torch.argmax(predictions, dim=1).item()
            fruit_labels = ['Apple', 'Avocado', 'Banana', 'Cherry', 'Kiwi', 
                            'Mango', 'Orange', 'Pineapple', 'Strawberries', 'Watermelon']
            predicted_fruit = fruit_labels[predicted_idx]

            # Display result
            st.success(f"**Predicted Fruit:** {predicted_fruit}")

    except Exception as error:
        st.error(f"Oops! Something went wrong: {error}")
