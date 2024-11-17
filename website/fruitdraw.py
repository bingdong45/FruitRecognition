import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import numpy as np

# Sidebar options for canvas
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=150,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

# Process the image data from the canvas
if canvas_result.image_data is not None:
    
    image_array = canvas_result.image_data.astype(np.uint8)  # Ensure data type is uint8
    torch_img = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0  # Normalize

    # Check the type and shape of the tensor
    st.write(f"Tensor Type: {type(torch_img)}")  # Should output: <class 'torch.Tensor'>
    st.write(f"Tensor Shape: {torch_img.shape}")  # Should output: [channels, height, width]