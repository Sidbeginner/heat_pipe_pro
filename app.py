import streamlit as st
import joblib
import pandas as pd
from utils.engine import HeatPipeRecommendationEngine # Assuming logic from previous step

# Page Config
st.set_page_config(page_title="Heat Pipe Pro", layout="wide")

# Sidebar Navigation
page = st.sidebar.radio("Navigate", ["Home", "About Project", "User Input", "Results", "Importance"])

# Initialize Engine
if 'engine' not in st.session_state:
    st.session_state.engine = HeatPipeRecommendationEngine()
    st.session_state.engine.load_and_train()

if page == "Home":
    st.title("Heat Pipe Recommendation Engine")
    st.subheader("High-Performance Thermal Management System")
    st.write("Optimize your cooling design with data-driven predictions.")
    # 

elif page == "About Project":
    st.title("About")
    st.write("This tool uses machine learning surrogate models to predict thermal resistance based on physical configuration.")

elif page == "User Input":
    st.title("Engineering Specifications")
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        load = col1.number_input("Heat Load (W)", 10.0, 500.0)
        area = col2.number_input("Heat Source Area (cm²)", 1.0, 50.0)
        amb = col1.number_input("Ambient Temp (°C)", 10.0, 50.0)
        cool = col2.selectbox("Cooling Method", ["Forced Convection", "Natural Convection"])
        ori = col1.selectbox("Orientation", ["Horizontal", "Vertical (Bottom Heat)"])
        max_l = col2.number_input("Max Length (mm)", 50.0, 500.0)
        max_d = col1.number_input("Max Diameter (mm)", 3.0, 20.0)
        
        submitted = st.form_submit_button("Generate Recommendations")
        if submitted:
            st.session_state.results = st.session_state.engine.recommend(load, area, amb, cool, ori, max_l, max_d)
            st.success("Analysis Complete!")

elif page == "Results":
    st.title("Top 3 Configurations")
    if 'results' in st.session_state:
        for rec in st.session_state.results:
            with st.expander(f"Rank {rec['Rank']}: {rec['Heat Pipe Type']}"):
                st.json(rec)
    else:
        st.warning("Please submit input form first.")

elif page == "Importance":
    st.title("Feature Importance")
    st.write("Understanding which physical parameters drive thermal performance.")
    #
