import streamlit as st
import torch
import numpy as np
from model import CNN
from torchvision import transforms
import io
from PIL import Image

st.title("_What's that_ :green[Fruit] :apple:")
st.header(":arrow_up_small: Mode")
st.subheader("Directions")
multi = '''
1. Click on browse files
2. Select a picture of fruit you want to identify
3. Then scroll down!
'''
st.markdown(multi)
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(bytes_data)).convert('RGB')


        st.image(image, caption='Uploaded Image')

    
        transform = transforms.Compose([
            transforms.Resize((250, 250)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                                 std=[0.229, 0.224, 0.225])
        ])


        torch_img = transform(image)
        torch_img = torch_img.unsqueeze(0)  


        model = CNN()
        state_dict = torch.load('./model_weight.pth', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        model.eval()

        with torch.no_grad():
            result = model(torch_img)
            predicted_class = torch.argmax(result, dim=1).item()
            labels = ['Apple', 'Avocado', 'Banana', 'Cherry', 'Kiwi', 'Mango', 'Orange', 'Pineapple', 'Strawberries', 'Watermelon']
            fruit = labels[predicted_class]
            st.header(f"Predicted Fruit: {fruit}")

        # st.write(f"Tensor Type: {type(torch_img)}")
        # st.write(f"Tensor Shape: {torch_img.shape}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
