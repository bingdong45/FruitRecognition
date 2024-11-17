import streamlit as st
import torch
import numpy as np

st.title("About Us :star2:")
st.subheader("This website allows users to input an image of a fruit and it will identify and return the type of fruit depicted in the image. We are a group of students studying Computer Science at the University of Wisconsin - Madison. Our purpose is to educate youngsters using technologies in various ways. This fruit identifier is our first steps towards our goal and we are dedicated to working on other similar projects in the future.")

multi = '''
Rate our program
'''

st.markdown(multi)

selected = st.feedback("stars")

