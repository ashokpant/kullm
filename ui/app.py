"""
-- Created by: Ashok Kumar Pant
-- Email: asokpant@gmail.com
-- Created on: 03/04/2025
"""

import requests
import streamlit as st

from ui.api_config import APIConfig

st.title("Surname Classification")

surname = st.text_input("Enter a surname:")
if st.button("Classify"):
    if surname:
        response = requests.post(APIConfig.CLASSIFY_SURNAME, json={"query": surname})
        if response.status_code == 200:
            result = response.json()
            st.write(f"Category: {result['category']}")
        else:
            st.write("Error: Unable to classify the surname.")
    else:
        st.write("Please enter a surname.")
