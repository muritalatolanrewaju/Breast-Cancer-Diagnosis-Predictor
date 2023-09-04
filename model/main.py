# Using Scikit-learn to Build and Train Machine Learning Models

import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Model creation function
def create_model(data):
    # Split data into predictors(X) and target variable(y)
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Scale data to the same range
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)

    # Print model accuracy and classification report
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return model, scaler


# Clean data function
def get_clean_data():
    data = pd.read_csv("../data/data.csv")

    # Drop Unnamed: 32 and id column
    data = data.drop(["Unnamed: 32", 'id'], axis=1)

    # Encode diagnosis column to 0 and 1 (Malignant(M) = 1 and Benign(B) = 0)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data


def main():
    data = get_clean_data()

    # Export model and scalar
    model, scalar = create_model(data)

    # Save model and scalar as a binary (pickle) file
    with open('../model/breast_cancer_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('../model/breast_cancer_scalar.pkl', 'wb') as f:
        pickle.dump(scalar, f)


if __name__ == "__main__":
    main()
