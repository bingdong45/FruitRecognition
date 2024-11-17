import streamlit as st
import torch
import numpy as np

st.title("_What's that_ :green[Fruit] :apple:")
st.header(":arrow_up_small: Mode")
st.subheader("Directions")
multi = '''
1. Click on upload box
2. Select a picture of fruit you want to identify
3. Click “Search” to see your results!
'''
st.markdown(multi)
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read image file buffer as a 3D uint8 tensor with PyTorch:
    bytes_data = uploaded_file.getvalue()
    torch_img = torch.ops.image.decode_image(
        torch.from_numpy(np.frombuffer(bytes_data, np.uint8)), 3
    )
    # Check the type of torch_img:
    # Should output: <class 'torch.Tensor'>
    st.write(type(torch_img))

    # Check the shape of torch_img:
    # Should output shape: torch.Size([channels, height, width])
    st.write(torch_img.shape)