import streamlit as st
import torch
import numpy as np

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
    # To read image file buffer as a 3D uint8 tensor with PyTorch:
    bytes_data = img_file_buffer.getvalue()
    torch_img = torch.ops.image.decode_image(
        torch.from_numpy(np.frombuffer(bytes_data, np.uint8)), 3
    )

    # Check the type of torch_img:
    # Should output: <class 'torch.Tensor'>
    st.write(type(torch_img))

    # Check the shape of torch_img:
    # Should output shape: torch.Size([channels, height, width])
    st.write(torch_img.shape)