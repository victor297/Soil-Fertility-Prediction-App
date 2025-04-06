import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Define Classifier
class SoilQualityClassifier:
    def __init__(self):
        with open('random_forest_pkl.pkl', 'rb') as file:
            self.model = pickle.load(file)

    def preprocess(self, input_data):
        df = pd.DataFrame([input_data])
        df_log = df.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x)
        return df_log

    def predict(self, input_data):
        df_processed = self.preprocess(input_data)
        prediction = self.model.predict(df_processed)[0]
        return prediction

    def interpret_result(self, prediction):
        classes = ["Less Fertile", "Fertile", "Highly Fertile"]
        return classes[prediction]

# Streamlit UI
st.set_page_config(page_title="🌱 Soil Fertility Predictor", layout="centered")
st.title("🌾 Soil Fertility Prediction App")
st.write("Project By: Odebamire Ismail Abimbola 22D/7HCS/417")
st.write("Provide elemental soil data below:")

# Feature inputs
features = {
    "N": st.number_input("Nitrogen (N)", min_value=0.0, format="%.4f"),
    "P": st.number_input("Phosphorous (P)", min_value=0.0, format="%.4f"),
    "K": st.number_input("Potassium (K)", min_value=0.0, format="%.4f"),
    "ph": st.number_input("Soil pH", min_value=0.0, format="%.2f"),
    "ec": st.number_input("Electrical Conductivity (EC)", min_value=0.0, format="%.4f"),
    "oc": st.number_input("Organic Carbon (OC)", min_value=0.0, format="%.4f"),
    "S": st.number_input("Sulfur (S)", min_value=0.0, format="%.4f"),
    "zn": st.number_input("Zinc (Zn)", min_value=0.0, format="%.4f"),
    "fe": st.number_input("Iron (Fe)", min_value=0.0, format="%.4f"),
    "cu": st.number_input("Copper (Cu)", min_value=0.0, format="%.4f"),
    "Mn": st.number_input("Manganese (Mn)", min_value=0.0, format="%.4f"),
    "B": st.number_input("Boron (B)", min_value=0.0, format="%.4f"),
}

if st.button("Predict Fertility"):
    try:
        classifier = SoilQualityClassifier()
        result = classifier.predict(features)
        interpreted = classifier.interpret_result(result)
        st.success(f"🌾 The soil is **{interpreted}**.")
    except Exception as e:
        st.error(f"🚫 Error during prediction: {str(e)}")
