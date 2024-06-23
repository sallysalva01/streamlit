#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the data
file_path = r'C:\Users\SaChauke\Downloads\HEARTS.csv'
data = pd.read_csv(file_path, delimiter=';')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Identify features and target
X = data.drop(columns=['target'])
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numerical features (scaling and imputing missing values)
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features (encoding and imputing missing values)
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Display the shape of the preprocessed data
print("Shape of preprocessed training data:", X_train_preprocessed.shape)
print("Shape of preprocessed testing data:", X_test_preprocessed.shape)


# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load the data
file_path = r'C:\Users\SaChauke\Downloads\HEARTS.csv'
data = pd.read_csv(file_path, delimiter=';')

# Identify features and target
X = data.drop(columns=['target'])
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numerical features (scaling and imputing missing values)
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features (encoding and imputing missing values)
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create pipelines for each model
log_reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

# Fit and evaluate each model
pipelines = [log_reg_pipeline, rf_pipeline, svm_pipeline]
model_names = ['Logistic Regression', 'Random Forest', 'SVM']
best_model = None
best_accuracy = 0

for model, name in zip(pipelines, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy:.4f}')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Save the best model to disk
joblib.dump(best_model, 'best_model.pkl')
print(f'Best model saved: {best_model.steps[-1][0]} with accuracy {best_accuracy:.4f}')


# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load the data
file_path = r'C:\Users\SaChauke\Downloads\HEARTS.csv'
data = pd.read_csv(file_path, delimiter=';')

# Identify features and target
X = data.drop(columns=['target'])
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numerical features (scaling and imputing missing values)
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features (encoding and imputing missing values)
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline that includes preprocessing and the SVM model
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(probability=True, random_state=42))
])

# Train the SVM model
svm_pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Accuracy: {accuracy:.4f}')

# Save the model to disk
model_save_path = r'C:\Users\SaChauke\anaconda3\Lib\site-packages\streamlit\best_model.pkl'
joblib.dump(svm_pipeline, model_save_path)
print(f'Model saved to {model_save_path}')


# In[4]:


import os

import joblib
 
# Define the model save path

model_save_path = 'C:\\Users\\Sachauke\\anaconda3\\Lib\\site-packages\\streamlit\\best_model.pkl'
 
# Ensure the directory exists

os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
 
# Save the model to disk

joblib.dump(svm_pipeline, model_save_path)

print(f'Model saved to {model_save_path}')


# In[5]:


import streamlit as st
import joblib
import numpy as np 
import pandas as pd

# Load the trained model
model = joblib.load('C:\\Users\\Sachauke\\anaconda3\\Lib\\site-packages\\streamlit\\best_model.pkl')

# Define the app
st.title('Heart Disease Prediction')
st.write('Enter the patient details to predict the likelihood of heart disease.')

# Input fields for patient data
age = st.number_input('Age', min_value=1, max_value=120, value=30)
sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
cp = st.number_input('Chest Pain Type', min_value=0, max_value=3, value=0)
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=120)
chol = st.number_input('Serum Cholesterol (chol)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
restecg = st.number_input('Resting Electrocardiographic Results (restecg)', min_value=0, max_value=2, value=1)
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.number_input('Slope of the Peak Exercise ST Segment (slope)', min_value=0, max_value=2, value=1)
ca = st.number_input('Number of Major Vessels Colored by Flouroscopy (ca)', min_value=0, max_value=4, value=0)
thal = st.number_input('Thalassemia (thal)', min_value=0, max_value=3, value=2)

# Convert inputs into a dataframe
input_data = pd.DataFrame({
    'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol], 'fbs': [fbs],
    'restecg': [restecg], 'thalach': [thalach], 'exang': [exang], 'oldpeak': [oldpeak],
    'slope': [slope], 'ca': [ca], 'thal': [thal]
})

# Predict and display result
if st.button('Predict'):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]  # Probability of positive class

    if prediction == 1:
        st.error(f'The patient is likely to have heart disease. Probability: {prediction_proba:.2f}')
    else:
        st.success(f'The patient is not likely to have heart disease. Probability: {prediction_proba:.2f}')

# Add some information about the app
st.write("""
### About the App
This application uses a machine learning model to predict the likelihood of heart disease based on patient data. Please fill in all the fields with the patient's information to get a prediction.
""")


# In[ ]:




