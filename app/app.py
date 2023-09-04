# This is the main file of the Streamlit application
import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


# Get clean data function for the sidebar
def get_clean_data():
    data = pd.read_csv("../data/data.csv")

    # Drop Unnamed: 32 and id column
    data = data.drop(["Unnamed: 32", 'id'], axis=1)

    # Encode diagnosis column to 0 and 1 (Malignant(M) = 1 and Benign(B) = 0)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data


# Sidebar function
def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    # Call get_clean_data function
    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    # Create a dictionary of input key value pairs
    input_dict = {}

    # Loop through the slider labels list and create a min, max, and mean slider for each
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label=label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict


# Add function to scale input values
def get_scaled_values(input_dict):
    data = get_clean_data()

    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict


# Add radar chart function
def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                  'Smoothness', 'Compactness',
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )
    return fig


# Add prediction function to get prediction from model
def add_predictions(input_data):
    model = pickle.load(open("../model/breast_cancer_model.pkl", "rb"))
    scalar = pickle.load(open("../model/breast_cancer_scalar.pkl", "rb"))

    # Convert input data to array
    input_array = np.array(list(input_data.values())).reshape(1, -1)

    # Scale input data
    input_array_scaled = scalar.transform(input_array)

    # Fit scaled value to prediction model
    prediction = model.predict(input_array_scaled)

    st.subheader("Cell Cluster Prediction")
    st.write("The tissue sample is : ")

    # Write prediction
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malignant'>Malignant</span>", unsafe_allow_html=True)

    st.write("Probability of being Benign is: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being Malignant is: ", model.predict_proba(input_array_scaled)[0][1])

    st.write("This app is designed to aid medical professionals in the diagnostic process but should not serve as a "
             "substitute for a professional diagnosis.")


def main():
    st.set_page_config(
        page_title="Breast Cancer Diagnosis Predictor",
        page_icon=":hospital:", layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add CSS Styles
    with open("../assets/style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    # Add sidebar data to dictionary
    input_data = add_sidebar()

    # Add a title and some text to the app:
    with st.container():
        st.title("Breast Cancer Diagnosis Predictor")
        st.write("This machine learning model evaluates tissue samples to determine the nature of a breast mass. "
                 "Users have the option to manually adjust the measurements through easy-to-use sliders located on "
                 "the sidebar. This app can be linked with a cytology lab to facilitate quick and accurate breast "
                 "cancer assessments.")

    # Create Radar Chart and prediction columns at 4:1 ratio
    col1, col2 = st.columns([4, 1])

    # Write information in the first column
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    # Write information in the second column
    with col2:
        add_predictions(input_data)


if __name__ == "__main__":
    main()
