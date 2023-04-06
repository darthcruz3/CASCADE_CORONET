# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:46:44 2023

@author: ldcruz
"""


# load and evaluate a saved model
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import streamlit as st
from sklearn.metrics import accuracy_score
from numpy import loadtxt
import pandas as pd

import pickle

filename = 'LDAmodel_timepoint2_LGD.sav'  # specify the saved filename

# Restore all objects and model stored in tuple pickle file
LDA_model_2, X, y, df_biomarkers_deteriorators_timepoint2, X_train_smote, y_train_smote, X_test, y_test, y_pred_class_on_test = pickle.load(
    open(filename, 'rb'))


print("Accuracy:", metrics.accuracy_score(y_test, y_pred_class_on_test))


# Get the feature input from the user

def get_user_input():

    IP10 = st.sidebar.slider('IP10', 1.00, 8000.00, 679.37)
    IL27 = st.sidebar.slider('IL27', 1.00, 5600.00, 4865.55)
    Ferritin = st.sidebar.slider('Ferritin', 1.00, 500.00, 12.00)
    MDC = st.sidebar.slider('MDC', 1.00, 1500.00, 287.35)
    CRP = st.sidebar.slider('CRP', 0.00, 30.00, 1.0)
    ComplementC5 = st.sidebar.slider('ComplementC5', 0.00, 250.00, 55.00)

    # Store a dictionary into a variable
    user_data = {'IP10': IP10,
                 'IL27': IL27,
                 'Ferritin': Ferritin,
                 'MDC': MDC,
                 'CRP': CRP,
                 'ComplementC5': ComplementC5
                 }

    # Transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features


# store the user's input into a variable
user_input = get_user_input()

# set a subheader and display the users input
#st.subheader('User input:')
# so when user puts in a value, we can see it on the app
#st.write(user_input)

st.header('CASCADE/CORONET study')
st.subheader('Clinical deterioration risk stratification Machine-learning model in COVID-19')

st.write(' ')
st.write('A collaborative project between Portsmouth University Hospital, UK, Cincinnati Medical Centre, USA, Ohio State College of Medicine, USA and AKARI therapeutics Ltd.')
st.write(' ')
st.write(' ')
st.write('**probable clinical risk at ~80% accuracy suggests:**')
y_new = LDA_model_2.predict(user_input)
#st.write(y_new)

if y_new == 1:
    st.write("**within normal parameters**")
    
elif y_new == 2:
    st.write("**clinically stable**")
    
elif y_new == 3:
    st.write("**PROBABLE CLINICAL DETERIORATION**")

st.write('')
st.write('Note: this model is in development stage, not yet for approved clinical use')