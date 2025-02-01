
import streamlit as st
import pickle
import gzip
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge

# Función para cargar el modelo
def load_model():
    try:
        with gzip.open('model_trained_regressor.pkl.gz', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Función principal
def main():
    # Personalización del título
    st.markdown("""
        <h1 style='text-align: center; color: #2E86C1;'>Predicción de Precios de Viviendas en Boston</h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <p style='text-align: center; font-size:18px; color: #5D6D7E;'>
        Introduce las características de la casa para obtener una estimación de su precio.
        </p>
    """, unsafe_allow_html=True)
    
    # Imagen de una casa
    st.image("house.jpg", use_container_width=True, caption="Ejemplo de Vivienda")
    
    # Sección de entrada de datos con mejor visualización
    st.subheader("Características de la vivienda")
    
    crim = st.number_input("Tasa de criminalidad (CRIM)", min_value=0.0, format="%.4f")
    zn = st.number_input("Proporción de terreno residencial (ZN)", min_value=0.0, format="%.2f")
    indus = st.number_input("Proporción de acres de negocios (INDUS)", min_value=0.0, format="%.2f")
    chas = st.selectbox("Límite con el río Charles (CHAS)", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    nox = st.number_input("Concentración de óxidos de nitrógeno (NOX)", min_value=0.0, format="%.4f")
    rm = st.number_input("Número promedio de habitaciones (RM)", min_value=0.0, format="%.2f")
    age = st.number_input("Porcentaje de unidades antiguas (AGE)", min_value=0.0, format="%.2f")
    dis = st.number_input("Distancia a centros de empleo (DIS)", min_value=0.0, format="%.2f")
    rad = st.number_input("Índice de accesibilidad a autopistas (RAD)", min_value=0, format="%d")
    tax = st.number_input("Tasa de impuesto sobre la propiedad (TAX)", min_value=0, format="%d")
    ptratio = st.number_input("Proporción alumno-maestro (PTRATIO)", min_value=0.0, format="%.2f")
    b = st.number_input("Índice de población afroamericana (B)", min_value=0.0, format="%.2f")
    lstat = st.number_input("Porcentaje de población de estatus bajo (LSTAT)", min_value=0.0, format="%.2f")
    
    # Botón de predicción con estilo
    if st.button("🔍 Predecir Precio"):
        model = load_model()
        if model is not None:
            features = [[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]]
            prediction = model.predict(features)
            
            st.success(f"💰 El precio predicho de la casa es: ${prediction[0]:,.2f}")
            
            st.info("""
                **Hiperparámetros del mejor modelo:**
                - 🔹 alpha: 0.1  
                - 🔹 kernel: rbf
                
                **Evaluación de Modelos:**
                - ElasticNet con StandardScaler -> MAE: 3.4372 | Parámetros: {'reg__alpha': 0.1, 'reg__l1_ratio': 0.1}
                - KernelRidge con StandardScaler -> MAE: 2.6156 | Parámetros: {'reg__alpha': 0.1, 'reg__kernel': 'rbf'} ✅
                - ElasticNet con MinMaxScaler -> MAE: 3.4694 | Parámetros: {'reg__alpha': 0.1, 'reg__l1_ratio': 1.0}
                - KernelRidge con MinMaxScaler -> MAE: 2.8787 | Parámetros: {'reg__alpha': 0.1, 'reg__kernel': 'poly'}
                - ElasticNet sin escalado -> MAE: 3.5148 | Parámetros: {'reg__alpha': 0.1, 'reg__l1_ratio': 1.0}
                - KernelRidge sin escalado -> MAE: 3.5921 | Parámetros: {'reg__alpha': 1.0, 'reg__kernel': 'linear'}
                
                **📌 Mejor Modelo:** KernelRidge con StandardScaler
                - Pipeline: StandardScaler + KernelRidge(alpha=0.1, kernel='rbf')
                - MAE más bajo: 2.6156 🎯
            """)
    
    # Pie de página con el nombre del creador
    st.markdown("""
        <hr>
        <p style='text-align: center; font-size:16px; color: #7D7D7D;'>
        Página creada por <strong>Yulisa Ortiz Giraldo</strong>
        </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

