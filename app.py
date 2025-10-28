"""Main Dashboard"""

import os
import streamlit as st
from google import genai
from rag_core.pipeline import load_and_vectorize_data
from rag_core.generator import generate_rag_prompt
from rag_core.evaluator import evaluate_faithfulness

# --- 0. CONFIGURACIÓN E INICIALIZACIÓN DE ESTADO ---
st.set_page_config(layout="wide")
st.title("🏡 PropGenius: Generador de Prompts Inmobiliarios (RAG)")
st.markdown("### Arquitectura Modular: Ingesta, Generación y Testing.")

# Inicializar estados de sesión si no existen
if 'client' not in st.session_state:
    st.session_state.client = None
if 'vectorstore_db' not in st.session_state:
    st.session_state.vectorstore_db = None
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False

# --- 1. BARRA LATERAL PARA INPUT DE API KEY Y DATOS ---

st.sidebar.header("🔑 Configuración y Clave API")
API_KEY = st.sidebar.text_input(
    "Google AI API Key:", 
    type="password",
    help="Introduce tu clave. No se guarda."
)
st.sidebar.markdown("[Get a Gemini API key](https://aistudio.google.com/app/api-keys)")

if not st.session_state.api_key_set and API_KEY:
    try:
        # Intentar inicializar el cliente con la clave proporcionada
        os.environ["GOOGLE_API_KEY"] = API_KEY
        client = genai.Client(api_key=API_KEY)
        st.session_state.client = client
        st.session_state.api_key_set = True

        # Cargar la base de conocimiento vectorizada una vez que el cliente está listo
        with st.spinner('Cargando la Arquitectura RAG... (Simulación ETL)'):
            st.session_state.vectorstore_db = load_and_vectorize_data(client=client)

        st.sidebar.success("Conexión y RAG Pipeline listos. ¡Procede con los datos!")
        # Forzar un rerun para actualizar la UI principal
        st.rerun()

    except genai.errors.APIError as e:
        st.sidebar.error(f"Error al inicializar: Verifica tu clave API. Detalles: {e}")
        st.session_state.api_key_set = False
        st.session_state.client = None

# --- 2. LÓGICA DE LA APLICACIÓN PRINCIPAL ---

if not st.session_state.api_key_set:
    st.info("Introduce tu Google AI API Key en la barra lateral para comenzar.")
else:
    # Si todo está inicializado, mostramos el formulario
    st.sidebar.header("📝 Datos de la Propiedad")
    with st.sidebar.form("property_form"):
        propiedad_tipo = st.selectbox("Tipo de Propiedad", ["Casa", "Departamento", "Terreno"])
        propiedad_ubicacion = st.text_input("Ubicación", "Condesa, CDMX")
        propiedad_caracteristicas = st.text_area("Características Clave",
                                                 "3 recámaras, terraza, seguridad 24h.")
        propiedad_publico = st.selectbox("Tono/Público Objetivo",
                                         ["Familias con niños", "Ejecutivos jóvenes"])
        submitted = st.form_submit_button("✨ Generar Prompt y Evaluar")

    if submitted:
        if not all([propiedad_ubicacion, propiedad_caracteristicas]):
            st.warning("Por favor, completa la ubicación y las características.")
            st.stop()

        propiedad_data = {
            'type': propiedad_tipo,
            'location': propiedad_ubicacion,
            'charact': propiedad_caracteristicas,
            'audience': propiedad_publico
        }

        client = st.session_state.client
        vectorstore_db = st.session_state.vectorstore_db

        # 2.1 Ejecución del Pipeline RAG (Generator)
        with st.spinner('1. Ejecutando el ML Pipeline RAG...'):
            generated_copy, context_used, query_str = generate_rag_prompt(
                client, vectorstore_db, propiedad_data)

        st.success("✅ Prompt de Venta Generado:")
        st.markdown(generated_copy)

        st.divider()

        # 2.2 Ejecución del Testing (Evaluator)
        st.subheader("📊 Testing de Calidad del Output (Evaluación LLM)")

        col1, col2 = st.columns([1, 2])

        with col1:
            with st.spinner('2. Corriendo prueba de Fidelidad (Faithfulness)...'):
                evaluation_result = evaluate_faithfulness(
                    client, query_str, context_used, generated_copy)

            score = evaluation_result.get('fidelidad_score', 0)
            justification = evaluation_result.get('justification', 'No disponible.')

            if score == 1:
                st.metric(label="Fidelidad al Contexto Histórico",
                          value="ALTA ✅", delta="Previene Alucinaciones")
            else:
                st.metric(label="Fidelidad al Contexto Histórico",
                          value="BAJA ❌", delta="Posible Alucinación")

        with col2:
            st.info(f"**Justificación del LLM Evaluador (Gemini Pro):** {justification}")


        # 2.3 Documentación del Pipeline
        st.subheader("🔎 Documentación: Trazabilidad del Dato")
        with st.expander("Ver Documentos de Contexto Histórico (Data Lineage)"):
            st.caption("Datos pasados que el modelo usó como referencia (context):")
            st.code(context_used, language='text')
