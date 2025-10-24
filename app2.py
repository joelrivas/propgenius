"""Main Dashboard"""

import os
import streamlit as st
import json
from google import genai
from google.genai.errors import APIError


# Connection
try:
    API_KEY = os.environ['GOOGLE_API_KEY']
    client = genai.Client(api_key=API_KEY)
except KeyError:
    st.error("Error: Gemini API Key not found.")
    st.stop()
except APIError as e:
    st.error(f"Error starting Gemini Client: {e}")
    st.stop()
