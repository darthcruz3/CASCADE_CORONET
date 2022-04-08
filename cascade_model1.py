# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:28:09 2022

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

filename = 'LDAmodel_timepoint1.sav'  # specify the saved filename

# Restore all objects and model stored in tuple pickle file
LDA_model_1, X, y, df_biomarkers_deteriorators_timepoint1, X_train_smote, y_train_smote, X_test, y_test, y_pred_class_on_test = pickle.load(
    open(filename, 'rb'))


print("Accuracy:", metrics.accuracy_score(y_test, y_pred_class_on_test))


# Get the feature input from the user

def get_user_input():

    LDH = st.sidebar.slider('LDH', 1, 3000, 2084)
    IL27 = st.sidebar.slider('IL27', 1.00, 8600.00, 4865.55)
    RANTES = st.sidebar.slider('RANTES', 1.00, 1300.00, 486.89)
    MDC = st.sidebar.slider('MDC', 1.00, 1500.00, 287.35)
    Platelets = st.sidebar.slider('Platelets', 1, 500, 176)
    PDGFAA = st.sidebar.slider('PDGFAA', 1.00, 1500.00, 360.66)
    MIGCXCL9 = st.sidebar.slider('MIGCXCL9', 1.00, 9500.00, 2363.34)
    Ferritin = st.sidebar.slider('Ferritin', 1.00, 15000.00, 741.00)
    IP10 = st.sidebar.slider('IP10', 1.00, 8000.00, 679.37)
    PDGFABBB = st.sidebar.slider('PDGFABBB', 1.00, 12500.00, 8230.59)

    # Store a dictionary into a variable
    user_data = {'LDH': LDH,
                 'IL27': IL27,
                 'RANTES': RANTES,
                 'MDC': MDC,
                 'Platelets': Platelets,
                 'PDGFAA': PDGFAA,
                 'MIGCXCL9': MIGCXCL9,
                 'Ferritin': Ferritin,
                 'IP10': IP10,
                 'PDGFABBB': PDGFABBB
                 }

    # Transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features


# store the user's input into a variable
user_input = get_user_input()

# set a subheader and display the users input
st.subheader('User input:')
# so when user puts in a value, we can see it on the app
st.write(user_input)


y_new = LDA_model_1.predict(user_input)
#st.write(y_new)

if y_new == 1:
    st.write("this patient's biomarker profile is within normal parameters")
    
elif y_new == 2:
    st.write("this patient is likely to be clinically stable")
    
elif y_new == 3:
    st.write("This patient's biomarker profile suggests potential clinical deterioration")
