
import streamlit as st
import pickle
import gzip

# Función para cargar el modelo
def load_model():
    """Cargar el modelo y sus pesos desde el archivo model_weights.pkl."""
    filename = 'model_trained_regressor.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


# Función principal
def main():
    st.title("Predicción de Precios de Viviendas en Boston")
    st.write("Introduce las características de la casa para predecir su precio.")

    # Campos de entrada para las características
    crim = st.number_input("Tasa de criminalidad per cápita por ciudad (CRIM)")
    zn = st.number_input("Proporción de terreno residencial zonificado para lotes de más de 25,000 pies cuadrados (ZN)")
    indus = st.number_input("Proporción de acres de negocios no minoristas por ciudad (INDUS)")
    chas = st.number_input("Variable ficticia Charles River (1 si el tramo limita con el río; 0 en caso contrario) (CHAS)")
    nox = st.number_input("Concentración de óxidos de nitrógeno (partes por 10 millones) (NOX)")
    rm = st.number_input("Número promedio de habitaciones por vivienda (RM)")
    age = st.number_input("Proporción de unidades ocupadas por el propietario construidas antes de 1940 (AGE)")
    dis = st.number_input("Distancias ponderadas a cinco centros de empleo de Boston (DIS)")
    rad = st.number_input("Índice de accesibilidad a autopistas radiales (RAD)")
    tax = st.number_input("Tasa de impuesto sobre la propiedad de valor total por $10,000 (TAX)")
    ptratio = st.number_input("Proporción alumno-maestro por ciudad (PTRATIO)")
    b = st.number_input("1000(Bk - 0.63)^2 donde Bk es la proporción de personas de ascendencia afroamericana por ciudad (B)")
    lstat = st.number_input("Porcentaje de población de estatus bajo (LSTAT)")

    # Botón para realizar la predicción
    if st.button("Predecir Precio"):
        model = load_model()
        features = [[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]]
        prediction = model.predict(features)
        st.success(f"El precio predicho de la casa es: ${prediction[0]:,.2f}")

        # Mostrar hiperparámetros del mejor modelo
        st.write("Hiperparámetros del mejor modelo:")
        st.write(model.get_params())

if __name__ == "__main__":
    main()
