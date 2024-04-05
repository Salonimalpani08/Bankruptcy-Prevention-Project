# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:30:54 2024

@author: dipen
"""

import pickle
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# Load the model
load = open('Deployment/rnd.pkl', 'rb')
model = pickle.load(load)

# Prediction function
def predict(Industrial_Risk, Management_Risk, Financial_Flexibility, Credibility, Competitiveness, Operating_Risk, new_Class):
    prediction = model.predict([[Industrial_Risk, Management_Risk, Financial_Flexibility, Credibility, Competitiveness, Operating_Risk]])
    return prediction[0]  # Use prediction[0] to get the actual prediction value

# Visualization function
def visualize():
    st.subheader('Interactive Line Chart: Example')

    # Generate example data for visualization
    bankruptcy = pd.DataFrame({
        'X': range(1, 11),
        'Y': np.random.rand(10)
    })

    # Create an interactive line chart using Altair
    chart = alt.Chart(bankruptcy).mark_line().encode(
        x='X',
        y='Y'
    ).interactive()

    # Display the chart
    st.altair_chart(chart, use_container_width=True)

# Main function
def main():
    st.title('Bankruptcy Prevention Project🧬')

    # Create tabs
    tabs = ["Prediction", "Visualization"]
    selected_tab = st.sidebar.radio("Select Option", tabs)

    # Content for each tab
    if selected_tab == "Prediction":
        st.markdown('This is a Random Forest machine learning model to predict Bankruptcy or Not... 🦾')

        Industrial_Risk = st.selectbox('Industrial_Risk:', [0, 0.5, 1], key='unique_key_6')
        Management_Risk = st.selectbox('Management_Risk:', [0, 0.5, 1], key='unique_key_1')
        Financial_Flexibility = st.selectbox('Financial_Flexibility:', [0, 0.5, 1], key='unique_key_2')
        Credibility = st.selectbox('Credibility:', [0, 0.5, 1], key='unique_key_3')
        Competitiveness = st.selectbox('Competitiveness:', [0, 0.5, 1], key='unique_key_4')
        Operating_Risk = st.selectbox('Operating_Risk:', [0, 0.5, 1], key='unique_key_5')

        if st.button('Predict'):
            Result = predict(Industrial_Risk, Management_Risk, Financial_Flexibility, Credibility, Competitiveness, Operating_Risk, 0)
            if Result == 0:
                st.success("Bankruptcy")
            else:
                st.success("Non-Bankruptcy")

    elif selected_tab == "Visualization":
        visualize()

if __name__ == '__main__':
    main()
