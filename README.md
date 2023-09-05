# Breast Cancer Diagnosis Predictor

## Introduction

The Breast-Cancer-Diagnosis-Predictor repository contains an app designed to assist healthcare professionals in early
breast cancer diagnosis. Utilizing a machine learning model trained with Logistic Regression, the app predicts whether a
breast mass is benign or malignant based on input measurements. It also features a radar chart for visual data
representation and displays the predicted diagnosis along with its associated probability.

## Features

### Data Input

- **Manual Entry**: Users have the option to manually input measurements related to the breast mass.
- **Lab Connection**: The app is designed for potential integration with cytology labs to automate data input directly
  from lab machines. (Note: The actual lab connection is not part of the app itself.)

### Data Visualization

- **Radar Chart**: The app employs a radar chart to visually represent the input measurements, offering a quick and
  intuitive understanding of the data.

### Prediction and Probability

- **Diagnosis Prediction**:  The app employs a Logistic Regression algorithm to predict whether the breast mass is
  benign or malignant.
- **Probability Display**: Alongside the diagnosis, the app also displays the probability of the mass being benign or
  malignant.

## Technology Stack

- **Machine Learning Algorithm**:  The model is trained using Logistic Regression on
  the [Breast Cancer Wisconsin (Diagnostic) Data Set.](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
  The model is trained using the Logistic Regression algorithm.
- **Visualization Tools**: Radar charts are used for data visualization.

## Check out the app

Here is the link to the
app: [Breast Cancer Diagnosis Predictor](https://breast-cancer-diagnosis-predictor-cfyc3bglmwsj8kbqhckal8.streamlit.app/)

## Installation

To run the Breast Cancer Diagnosis app locally, ensure you have Python 3.6 or higher installed. Install the required
packages by executing:

- Clone the repository

```bash
git clone https://github.com/muritalatolanrewaju/Breast-Cancer-Diagnosis-Predictor.git
```

- Change directory to the app folder

```bash
cd Breast-Cancer-Diagnosis-Predictor
```

- Install the required packages

```bash
pip install -r requirements.txt
```

This will install the necessary dependencies, including:

- streamlit
- numpy
- pandas
- pickle
- plotly
- scipy
- matplotlib
- scikit-learn

## Usage

To run the app, execute:

```bash
python3 -m streamlit run app.py
```

This will open the app in your default browser. You can then enter the measurements related to the breast mass and click
the "Predict" button to view the diagnosis and probability.

## Disclaimer

This app is developed as a machine learning exercise using the public dataset Breast Cancer Wisconsin (Diagnostic) Data
Set. It is intended solely for educational purposes in the field of machine learning and is not designed for
professional medical use. The dataset may not be entirely reliable for clinical diagnosis.

## Future Enhancements

- Integration with more advanced machine learning models for improved accuracy.
- Addition of other visualization tools for better data interpretation.
- Enabling real-time data input from cytology labs.

## Conclusion

The Breast-Cancer-Diagnosis-Predictor app demonstrates the potential of machine learning in healthcare for early breast
cancer diagnosis. While it is not intended for professional medical use, it provides valuable insights into the
capabilities of machine learning in medical diagnostics.