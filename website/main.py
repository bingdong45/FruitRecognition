import streamlit as st


pages = {
    "What's that fruit?": [
        st.Page("fruitvideo.py", title="Take Picture"),
        st.Page("fruitimage.py", title="Upload image"),
        st.Page("fruitdraw.py", title="Draw image")
    ],
    "Resources": [
        st.Page("About.py", title="About us"),
    ],
}

pg = st.navigation(pages)
pg.run()