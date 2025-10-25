"""Main Dashboard"""

import os
import streamlit as st
from google import genai
from google.genai.errors import APIError
from rag_core.pipeline import load_and_vectorize_data
from rag_core.generator import generate_rag_prompt
from rag_core.evaluator import evaluate_faithfulness

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

# Load vectorized data once (ETL simulation)
with st.spinner("Cargando la Arquitectura RAG...)"):
    try:
        vectorstore_db = load_and_vectorize_data(client=client)
    except FileNotFoundError:
        st.error("Error: Data files not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Streamlit App
st.title("PropGenius: Generadot de Prompts Inmobiliarios con RAG y Gemini AI")
st.markdown("### Arquitectura Modular: Ingesta, Generacion y Evaluacion.")

st.sidebar.header("Datos de la Propiedad")
with st.sidebar.form("prop_feat"):
    prop_type = st.selectbox("Tipo de Propiedad", ["Casa", "Departamento", "Terreno"])
    prop_location = st.text_input("Ubicación", "Providencia, Guadalajara, CDMX...")
    prop_charac = st.text_area("Características", "3 recámaras, 2 baños, 150m2...")
    prop_audience = st.selectbox("Tono/Publico Objetivo", ["Familia con niños", "Jóvenes profesionales", "Inversionistas"])
    submitted = st.form_submit_button("Generar Prompt y Evaluar")

if submitted:
    prop_feat = {
        "type": prop_type,
        "location": prop_location,
        "charact": prop_charac,
        "audience": prop_audience
    }

    with st.spinner("1. Ejecutando el Pipeline RAG..."):
        generated_copy, context_used, query_str = generate_rag_prompt(
            client=client,
            vectorstore=vectorstore_db,
            prop_feat=prop_feat
        )
    st.success("Prompt de Ventas Generado:")
    st.markdown(generated_copy)

    st.divider()
    st.subheader("2. Evaluación de Fidelidad del Prompt Generado (Evaluacion LLM)")
    col1, col2 = st.columns([1, 2])

    with col1:
        with st.spinner("Corriendo prueba de Fidelidad (Faithfulness)..."):
            evaluation_result = evaluate_faithfulness(client=client,
                                                      query=query_str,
                                                      context=context_used,
                                                      response=generated_copy)
        score = evaluation_result.get('fidelidad_score', 0)
        justification = evaluation_result.get('justificacion', 'No disponible.')

        if score == 1:
            st.metric(label="Fidelidad al Contexto Histórico", value="ALTA ✅", delta="Previene Alucinaciones")
        else:
            st.metric(label="Fidelidad al Contexto Histórico", value="BAJA ❌", delta="Posible Alucinación")
            
    with col2:
        st.info(f"**Justificación del LLM Evaluador (Gemini Pro):** {justification}")


    # 4. Documentación del Pipeline
    st.subheader("🔎 Documentación: Trazabilidad del Dato")
    with st.expander("Ver Documentos de Contexto Histórico (Data Lineage)"):
        st.caption("Estos son los datos pasados ('clean datasets') que el modelo usó como referencia de marca:")
        st.code(context_used, language='text')