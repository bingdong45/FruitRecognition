import streamlit as st
import torch
import numpy as np
from model import GoogleNet
from torchvision import transforms
import torch.nn.functional as F
import io
from PIL import Image

st.title("_What's that_ :green[Fruit] :apple:")
st.header(":camera: Mode")
st.subheader("Directions")
multi = '''
1. Click on the camera icon to use your camera  
2. Aim your camera at the fruit you want to identify and click “Take a Photo”  
3. Click “Search” to see your results!  
'''
st.markdown(multi)

img_file_buffer = st.camera_input("Take a picture here")

if img_file_buffer is not None:
    try:

        bytes_data = img_file_buffer.getvalue()
        image = Image.open(io.BytesIO(bytes_data)).convert('RGB')

        st.image(image, caption='Uploaded Image', use_column_width=True)

 
        transform = transforms.Compose([
            transforms.Resize((250, 250)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])


        torch_img = transform(image)
        torch_img = torch_img.unsqueeze(0)  # Add batch dimension


        model = GoogleNet()
        state_dict = torch.load('./model_weight.pth', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        model.eval()


        with torch.no_grad():
            result = model(torch_img)
            predicted_class = torch.argmax(result, dim=1).item()
            labels = ['Apple', 'Avocado', 'Banana', 'Cherry', 'Kiwi', 'Mango', 'Orange', 'Pineapple', 'Strawberries', 'Watermelon']
            fruit = labels[predicted_class]
            st.write(f"Predicted Fruit: {fruit}")

        # st.write(f"Tensor Type: {type(torch_img)}")
        # st.write(f"Tensor Shape: {torch_img.shape}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

