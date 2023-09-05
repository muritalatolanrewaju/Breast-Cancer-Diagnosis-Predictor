# This is the main file of the Streamlit application
import pickle

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans


# Get clean data function for the sidebar
def get_clean_data():
    data = pd.read_csv("data/data.csv")

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


# Descriptive function to plot K-means clusters
def get_kmeans_clusters():
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    # Initialize KMeans with 2 clusters
    km = KMeans(n_clusters=2, n_init=10)  # Set n_init explicitly to 10
    km.fit(X)

    # Create a custom colormap
    cmap = mcolors.ListedColormap(['red', 'green'])

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(x=X.iloc[:, 0], y=X.iloc[:, 1], c=km.labels_, cmap=cmap)

    # Add a colorbar
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.set_label('Cluster Labels')

    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])

    return fig


# Descriptive function to get k-means cluster plot
def get_kmeans_plot():
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    inertias = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    kmeans_plot, ax = plt.subplots()
    ax.plot(range(1, 11), inertias, marker='o')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia')

    return kmeans_plot


# Descriptive function to plot Hierarchical Dendrogram
def plot_hierarchical_dendrogram():
    # Get the clean data
    data = get_clean_data()

    # Drop the diagnosis column to perform clustering
    X = data.drop(['diagnosis'], axis=1)

    # Take a subset of the data for quicker computation
    X_subset = X.sample(frac=0.05, random_state=42)

    # Generate the linkage matrix
    Z = linkage(X_subset, 'ward')

    # Create the dendrogram
    fig, ax = plt.subplots(figsize=(15, 7))
    dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)

    ax.set_title('Hierarchical Clustering Dendrogram (Subset)')
    ax.set_xlabel('Data Points')
    ax.set_ylabel('Euclidean Distance')

    return fig


# Add function to get pie chart
def get_pie_chart():
    df = pd.read_csv("data/data.csv")
    labels = df.diagnosis.unique()
    values = [len(df[df.diagnosis == 'M']), len(df[df.diagnosis == 'B'])]
    trace = [go.Pie(labels=labels, values=values,
                    marker=dict(colors=["red", "green"]))]

    layout = go.Layout(title="Percentage of M = malignant, B = benign ")
    pie_chart = go.Figure(data=trace, layout=layout)
    return pie_chart


# Add function to get bar chart
def get_bar_chart():
    df = pd.read_csv("data/data.csv")
    trace = [go.Bar(x=df.diagnosis.unique(), y=(len(df[df.diagnosis == 'M']), len(df[df.diagnosis == 'B'])),
                    marker=dict(color=["red", "green"]))]

    layout = go.Layout(title="Diagnosis Count of M = malignant, B = benign ")
    bar_chart = go.Figure(data=trace, layout=layout)
    return bar_chart


# Add function to get scatter plot
def get_scatter_plot():
    df = pd.read_csv("data/data.csv")
    scatter_plot_1 = px.scatter(df, x="radius_mean", y="compactness_mean", color="diagnosis",
                                title="Scatter Plot of Radius Mean and Compactness Mean", width=800, height=800,
                                template='plotly_white', color_discrete_sequence=["red", "green"], hover_data=['id'])

    scatter_plot_2 = px.scatter(df, x="area_mean", y="radius_mean", color="diagnosis",
                                title="Scatter Plot of Radius Mean and Area Mean", width=800, height=800,
                                template='plotly_white', color_discrete_sequence=["red", "green"], hover_data=['id'])

    return scatter_plot_1, scatter_plot_2


# Add function to get distribution plot
def get_dist_plot():
    df = pd.read_csv("data/data.csv")
    hist_data_radius_mean = [df[df['diagnosis'] == 'M']['radius_mean'],
                             df[df['diagnosis'] == 'B']['radius_mean']]

    hist_data_texture_mean = [df[df['diagnosis'] == 'M']['texture_mean'],
                              df[df['diagnosis'] == 'B']['texture_mean']]

    group_labels = ['Malignant', 'Benign']

    dist_plot_radius_mean = ff.create_distplot(hist_data_radius_mean, group_labels, bin_size=0.2,
                                               colors=["red", "green"])
    dist_plot_texture_mean = ff.create_distplot(hist_data_texture_mean, group_labels, bin_size=0.2,
                                                colors=["red", "green"])

    dist_plot_radius_mean.update_layout(title_text='Distribution Plot of Radius Mean')
    dist_plot_texture_mean.update_layout(title_text='Distribution Plot of Texture Mean')

    return dist_plot_radius_mean, dist_plot_texture_mean


# Add function to get heatmap
def get_heatmap():
    heatmap = pd.read_csv("data/data.csv")
    heatmap = heatmap.drop(["Unnamed: 32", 'id'], axis=1)
    heatmap['diagnosis'] = heatmap['diagnosis'].map({'M': 1, 'B': 0})
    corr = heatmap.corr()

    heatmap = px.imshow(corr, color_continuous_scale='RdBu_r', title='Correlation Matrix', width=800, height=800,
                        range_color=[-1, 1], labels=dict(x="Features", y="Features", color="Correlation Coefficient"),
                        template='plotly_white', text_auto=".2f")
    return heatmap


# Add function to get descriptive statistics
def get_descriptive_stats():
    descriptive_stats = pd.read_csv("data/data.csv")
    descriptive_stats = descriptive_stats.drop(["Unnamed: 32", 'id'], axis=1)
    descriptive_stats['diagnosis'] = descriptive_stats['diagnosis'].map({'M': 1, 'B': 0})
    descriptive_stats = descriptive_stats.describe()
    descriptive_stats = descriptive_stats.transpose()
    return descriptive_stats


# Add prediction function to get prediction from model
def add_predictions(input_data):
    model = pickle.load(open("model/breast_cancer_model.pkl", "rb"))
    scalar = pickle.load(open("model/breast_cancer_scalar.pkl", "rb"))

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


# Add main function
def main():
    st.set_page_config(
        page_title="Breast Cancer Diagnosis Predictor",
        page_icon=":hospital:", layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add CSS Styles
    with open("assets/style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    # Add sidebar data to dictionary
    input_data = add_sidebar()

    # Add a title and some text to the app:
    with st.container():
        st.title("WGU Capstone: Breast Cancer Diagnosis Predictor")
        st.write("This machine learning model evaluates tissue samples to determine the nature of a breast mass. "
                 "Users have the option to manually adjust the measurements through easy-to-use sliders located on "
                 "the sidebar. This app can be linked with a cytology lab to facilitate quick and accurate breast "
                 "cancer assessments.")

    # Create Radar Chart and prediction columns at 4:1 ratio
    col1, col2 = st.columns([4, 1])

    # Write information in the first column
    with col1:
        # Add Radar Chart, K-Means Clustering, Pie Chart, Count Plot, Scatter Plot, Distribution Plot, Heatmap,
        # and Descriptive
        st.title("Data Visualization")

        st.subheader("Radar Chart")
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

        st.subheader("K-Means 2-Clusters")
        kmeans_clusters = get_kmeans_clusters()
        st.pyplot(kmeans_clusters)

        st.subheader("K-Means Cluster Plot")
        kmeans_plot = get_kmeans_plot()
        st.pyplot(kmeans_plot)

        st.subheader("Hierarchical Clustering Dendrogram")
        hierarchical_dendrogram_plot = plot_hierarchical_dendrogram()  # Descriptive Hierarchical Dendrogram
        st.pyplot(hierarchical_dendrogram_plot)

        st.subheader("Pie Chart")
        pie_chart = get_pie_chart()
        st.plotly_chart(pie_chart)

        st.subheader("Count Plot")
        bar_chart = get_bar_chart()
        st.plotly_chart(bar_chart)

        st.subheader("Scatter Plot")
        scatter_plot_1, scatter_plot_2 = get_scatter_plot()
        st.plotly_chart(scatter_plot_1)
        st.plotly_chart(scatter_plot_2)

        st.subheader("Distribution Plot")
        dist_plot_radius_mean, dist_plot_texture_mean = get_dist_plot()
        st.plotly_chart(dist_plot_radius_mean)
        st.plotly_chart(dist_plot_texture_mean)

        st.subheader("Heatmap")
        heatmap = get_heatmap()
        st.plotly_chart(heatmap)

        st.subheader("Descriptive Statistics")
        st.write(get_descriptive_stats())

    # Write information in the second column
    with col2:
        add_predictions(input_data)


if __name__ == "__main__":
    main()
