
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el modelo entrenado
def load_model():
    with open("model_trained_regressor.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Definir las 13 características
feature_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]

def main():
    st.title("Predicción del Precio de Casas - Boston Housing 🏡")
    st.markdown("Ingrese las características de la casa para obtener el precio estimado.")

    # Crear inputs para cada característica
    inputs = {}
    for feature in feature_names:
        inputs[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f")
    
    # Convertir a un array de numpy
    input_data = np.array(list(inputs.values())).reshape(1, -1)

    # Cargar modelo y predecir
    if st.button("Predecir Precio 💰"):
        model = load_model()
        prediction = model.predict(input_data)[0]
        st.success(f"Precio estimado de la casa: ${prediction:,.2f}")
    
    # Agregar visualización interactiva
    st.subheader("Distribución de Precios de las Casas")
    df = pd.read_csv("boston_housing.csv")  # Asegúrate de tener este archivo
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["MEDV"], bins=30, kde=True, color="blue")
    ax.axvline(prediction, color='red', linestyle='dashed', linewidth=2, label='Predicción')
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
