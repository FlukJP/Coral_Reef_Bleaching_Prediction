import streamlit as st

st.set_page_config(
    page_title="🏠 หน้าแรก",
    page_icon="🏠",
    layout="wide"
)

with st.sidebar:
    st.title("🪸 Coral Reef")
    st.divider()

    page = st.radio(
        "Menu",
        ["Home", "Model Info", "Prediction Machine Learning", "Prediction Neural Network", "Prediction Stacking Ensemble"],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("Coral Reef Bleaching Prediction")

if page == "Home":
    from pages.home import show
    show()

elif page == "Model Info":
    from pages.model_info import show
    show()

elif page == "Prediction Machine Learning":
    from pages.prediction_ML import show
    show()

elif page == "Prediction Neural Network":
    from pages.prediction_nn import show
    show()