import streamlit as st
import requests 
import base64
import os


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}

    /* Style for File Uploader */
    .stFileUploader {{
        background-color: rgba(255, 255, 255, 0.9) !important; /* Light, slightly transparent white */
        border-radius: 10px !important;
        padding: 10px !important;
    }}

    /* Style for Text Area */
    textarea {{
        background-color: rgba(255, 255, 255, 0.9) !important; /* Light, slightly transparent white */
        color: #333333 !important; /* Dark grey color */
        font-size: 16px;
        border-radius: 10px !important; /* Rounded corners */
        padding: 10px; /* Internal spacing */
    }}
    textarea:focus {{
        outline: none;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.3) !important; /* Subtle focus shadow */
    }}
    
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_png_as_page_bg('image.png')

# Exemple de contenu de l'application
st.title("MetroShield")
st.write("A robust solution ensuring safety and efficiency by detecting and addressing anomalies in metro operations.")

uploaded_file = st.file_uploader("Upload file", type=["csv"])

prediction_result = ""

BACKEND_URL = os.environ.get('BACKEND_URL', 'http://backend:8000')

if uploaded_file is not None:
    st.write("File uploaded successfully!")

    file_bytes = uploaded_file.read()

    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            files={"file": ("data.csv", file_bytes, "text/csv")}
        )

        # Si la requête réussit, afficher les prédictions
        if response.status_code == 200:
            result = response.json()
            prediction_result = f"{result['message']}\n\nPredictions:\n{result['predictions']}"
        else:
            prediction_result = f"Error from API: {response.json().get('message')}"
    except Exception as e:
        prediction_result = f"Failed to communicate with API: {str(e)}"

# Afficher le résultat de la prédiction dans un champ de texte
st.text_area("Prediction Results", prediction_result, height=300)
