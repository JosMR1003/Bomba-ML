# --- Importación de Librerías ---
import joblib  # Para cargar nuestros modelos guardados (.pkl)
import streamlit as st  # La librería principal para crear la aplicación web
import pandas as pd  # Para manejar los datos de entrada en un DataFrame
import altair as alt  # Para crear gráficos interactivos

# --- Configuración de la Página ---
# st.set_page_config() configura metadatos de la página, como el título y el ícono en la pestaña del navegador.
try:
    st.set_page_config(
        page_title="Failure Classifier",
        page_icon="icone.png",  # Ícono que aparecerá en la pestaña
        layout="wide"  # Usa el ancho completo de la página para el contenido
    )
except FileNotFoundError:
    # Si el ícono no se encuentra, configura la página sin él para evitar errores.
    st.set_page_config(
        page_title="Failure Classifier",
        layout="wide"
    )

# --- Carga de Modelos y Objetos de Preprocesamiento ---
# @st.cache_resource es un decorador de Streamlit muy importante.
# Evita que los modelos se recarguen desde el disco cada vez que el usuario interactúa con la app,
# haciendo que la aplicación sea mucho más rápida y eficiente.
@st.cache_resource
def load_models():
    try:
        # Cargamos los tres artefactos que guardamos en los notebooks anteriores.
        preprocessor = joblib.load('preprocessor_pipeline.pkl')
        model = joblib.load('final_model.joblib')
        label_encoder = joblib.load('label_encoder.pkl')
        return preprocessor, model, label_encoder
    except FileNotFoundError as e:
        # Manejo de error si los archivos del modelo no se encuentran en la ruta especificada.
        st.error(
            f"Error al cargar los archivos del modelo: {e}. "
        )
        return None, None, None
    except AttributeError as e:
        # Este error es común si la versión de scikit-learn con la que se entrenó el modelo
        # es diferente a la del entorno donde se ejecuta la app. El archivo requirements.txt ayuda a evitar esto.
        st.error(
            f"AttributeError: {e}\n\n"
            "Esto generalmente significa que hay una discrepancia de versiones de scikit-learn "
            "entre tu entorno de entrenamiento y este entorno de Streamlit. "
            "Por favor, crea un archivo requirements.txt y especifica la versión exacta "
            "de scikit-learn usada para el entrenamiento (ej., scikit-learn==1.2.2)."
        )
        return None, None, None

# Llamamos a la función para cargar los modelos.
preprocessor, model, label_encoder = load_models()

# --- Mapeos y Descripciones ---
# Un diccionario para mostrar descripciones amigables para cada tipo de fallo predicho.
FAILURE_DESCRIPTIONS = {
    'No Failure': "✅ La máquina está operando en condiciones normales. No se requiere mantenimiento inmediato.",
    'Heat Dissipation Failure': "🔥 La máquina se está sobrecalentando. Esto podría deberse a problemas con el sistema de refrigeración, altas temperaturas ambientales o una operación prolongada con alto torque. Revisa si hay obstrucciones en las ventilaciones y asegúrate de que el sistema de refrigeración funcione.",
    'Power Failure': "⚡ El modelo predice un fallo de potencia potencial. Esto a menudo se relaciona con caídas repentinas de torque o inconsistencias en la velocidad de rotación sin un desgaste correspondiente de la herramienta. Revisa la fuente de alimentación y las conexiones eléctricas.",
    'Overstrain Failure': "⚙️ La máquina está bajo una tensión excesiva, indicada por un alto torque combinado con una baja velocidad de rotación. Esto puede dañar los componentes. Reduce la carga de trabajo o busca obstrucciones mecánicas.",
    'Tool Wear Failure': "🔧 La herramienta está significativamente desgastada y necesita ser reemplazada. Este es un tipo de fallo común y se indica directamente por la métrica de 'Tool wear'."
}

# --- Barra Lateral para Entradas del Usuario ---
st.sidebar.header("⚙️ Parámetros de Entrada")

# Creamos los widgets (deslizadores y un menú desplegable) en la barra lateral para que el usuario ingrese los datos.
air_input = st.sidebar.slider('Temperatura del Aire [K]', min_value=290.0, max_value=310.0, value=300.0, step=0.1)
process_input = st.sidebar.slider('Temperatura del Proceso [K]', min_value=300.0, max_value=320.0, value=310.0, step=0.1)
rpm_input = st.sidebar.slider('Velocidad de Rotación [rpm]', min_value=1100, max_value=3000, value=1500, step=10)
torque_input = st.sidebar.slider('Torque [Nm]', min_value=3.0, max_value=80.0, value=40.0, step=0.1)
tool_wear_input = st.sidebar.slider('Desgaste de Herramienta [min]', min_value=0, max_value=260, value=100, step=1)
type_input = st.sidebar.selectbox('Tipo de Calidad de Máquina', options=['Low', 'Medium', 'High'])

# --- Interfaz Principal de la Aplicación ---
st.title('🛠️ Mantenimiento Predictivo: Clasificador de Fallos')
st.markdown(
    """
    Esta aplicación utiliza un modelo de machine learning para predecir fallos potenciales en equipos basándose en datos de sensores en tiempo real.
    Ajusta los parámetros en la barra lateral para ver cómo afectan la predicción de fallos.
    """
)
st.write("---")

# --- Función de Predicción ---
def prediction(air_temp, proc_temp, rotational_speed, torque_val, tool_wear_val, type_val):
    # Primero, verifica si los modelos se cargaron correctamente.
    if not all([preprocessor, model, label_encoder]):
        return None, None

    # Crea un diccionario con los datos de entrada del usuario.
    input_data = {
        'Air_temperature': [air_temp],
        'Process_temperature': [proc_temp],
        'Rotational_speed': [rotational_speed],
        'Torque': [torque_val],
        'Tool_wear': [tool_wear_val],
        'Type': [type_val]
    }
    # Convierte el diccionario a un DataFrame de pandas, ya que el pipeline de preprocesamiento espera uno.
    df_input = pd.DataFrame(input_data)
    # Aplica las transformaciones (escalado y one-hot encoding) a los datos de entrada.
    df_transformed = preprocessor.transform(df_input)
    # Usa el modelo para predecir las probabilidades de cada clase.
    prediction_proba = model.predict_proba(df_transformed)
    # Usa el modelo para predecir la clase más probable.
    prediction_class = model.predict(df_transformed)
    # Devuelve la clase predicha (como número) y las probabilidades.
    return prediction_class[0], prediction_proba

# --- Ejecución y Visualización de la Predicción ---
# Este bloque de código solo se ejecuta si el usuario hace clic en el botón de la barra lateral.
if st.sidebar.button('▶️ Predecir Tipo de Fallo', type="primary"):
    # Llama a la función de predicción con los valores de los widgets.
    predicted_class_num, prediction_confidence = prediction(
        air_temp=air_input,
        proc_temp=process_input,
        rotational_speed=rpm_input,
        torque_val=torque_input,
        tool_wear_val=tool_wear_input,
        type_val=type_input
    )

    # Si la predicción fue exitosa...
    if predicted_class_num is not None:
        # Usa el label_encoder para convertir la predicción numérica (ej. 2) de nuevo a su etiqueta de texto (ej. 'Power Failure').
        predicted_label = label_encoder.inverse_transform([predicted_class_num])[0]

        st.header("Resultado de la Predicción")
        # Dividimos la sección de resultados en dos columnas para una mejor organización.
        col1, col2 = st.columns([1, 2])

        with col1:
            # Muestra la predicción principal usando st.metric.
            st.metric(label="Fallo Predicho", value=predicted_label)
            st.write("**Descripción:**")
            # Muestra la descripción detallada del fallo predicho.
            st.info(FAILURE_DESCRIPTIONS.get(predicted_label, "No hay descripción disponible."))

        with col2:
            # Preparamos los datos de confianza para el gráfico.
            st.write("**Confianza de la Predicción**")
            confidence_df = pd.DataFrame(prediction_confidence, columns=label_encoder.classes_).T
            confidence_df = confidence_df.reset_index()
            confidence_df.columns = ['Tipo de Fallo', 'Confianza']

            # Creamos un gráfico de barras interactivo con Altair.
            chart = alt.Chart(confidence_df).mark_bar().encode(
                x=alt.X('Confianza:Q', axis=alt.Axis(format='%')), # Eje X: confianza, formateada como porcentaje.
                y=alt.Y('Tipo de Fallo:N', sort='-x'), # Eje Y: tipo de fallo, ordenado de mayor a menor confianza.
                tooltip=['Tipo de Fallo', alt.Tooltip('Confianza:Q', format='.2%')] # Tooltip que aparece al pasar el ratón.
            ).properties(
                title='Confianza del Modelo para Cada Tipo de Fallo'
            )
            # Mostramos el gráfico en la app.
            st.altair_chart(chart, use_container_width=True)

else:
    # Mensaje inicial que se muestra antes de que el usuario haga una predicción.
    st.info("Ajusta los parámetros en la barra lateral y haz clic en 'Predecir Tipo de Fallo' para ver un resultado.")

# --- Sección Explicativa ---
# st.expander crea una sección colapsable que el usuario puede abrir para obtener más detalles.
with st.expander("ℹ️ Sobre la Aplicación"):
    st.markdown("""
    **¿Cómo funciona?**

    1.  **Datos de Entrada:** Proporcionas las lecturas actuales de los sensores de la máquina utilizando los deslizadores y el menú desplegable de la izquierda.
    2.  **Preprocesamiento:** La aplicación toma tus datos brutos y los transforma (escalando valores numéricos y codificando los categóricos) para que el modelo pueda entenderlos.
    3.  **Predicción:** El modelo pre-entrenado de LightGBM (LGBM) analiza los datos transformados y calcula la probabilidad de cada tipo de fallo potencial.
    4.  **Resultado:** La aplicación muestra el tipo de fallo más probable y un gráfico que muestra la confianza del modelo en cada predicción.

    **Detalles del Modelo:**

    * **Tipo de Modelo:** `LightGBM Classifier`
    * **Propósito:** Clasificar diferentes tipos de fallos de máquinas basándose en datos de sensores.
    * **Características Utilizadas:** Temperatura del Aire, Temperatura del Proceso, Velocidad de Rotación, Torque, Desgaste de Herramienta y Tipo de Calidad de la Máquina.
    """)
