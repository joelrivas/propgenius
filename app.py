import streamlit as st
from google import genai

API_KEY = "AIzaSyCsM_TK4PgcJpmdDBSewaiU4BPgdYiLY_o"
client = genai.Client(api_key=API_KEY)

st.title("PropGenius: Generador de Prompts de Venta Inmobiliaria")
st.markdown("Genera descripciones de venta creativas y optimizadas usando IA.")

# Input del Usuario
property_type = st.selectbox(
    "Tipo de Propiedad",
    ["Casa", "Departamento", "Terreno", "Oficina", "Local Comercial"]
)
property_location = st.text_input("Ubicación (Ej: Condesa, CDMX o Sector Norte, Monterrey)")
property_chars = st.text_area("Características Clave (Ej: 3 recámaras, terraza, seguridad 24h, remodelado, vistas panorámicas)")
property_audience = st.selectbox(
    "Público Objetivo Principal",
    ["Familias con niños", "Ejecutivos jóvenes", "Inversionistas", "Retirados"]
)

if st.button("✨ Generar Prompt de Venta"):
    if not API_KEY:
        st.error("Por favor, configura tu API Key de Gemini.")
    elif not all([property_location, property_chars]):
        st.warning("Por favor, completa la ubicación y las características.")
    else:
        # Pasa a la generación
        with st.spinner('Pensando en el copy perfecto...'):
            prompt_generacion = f"""
            Eres un experto en bienes raíces y copywriter. Tu tarea es generar una descripción de venta corta, convincente y emocionalmente atractiva para una propiedad.

            **Información de la Propiedad:**
            - Tipo: {property_type}
            - Ubicación: {property_location}
            - Características: {property_chars}
            - Público Objetivo: {property_audience}

            **Requisitos de la Generación:**
            1. Debe tener un título llamativo (máx. 10 palabras).
            2. La descripción debe ser de 3-5 párrafos cortos.
            3. El tono debe ser aspiracional y persuasivo.
            4. Debe resaltar un beneficio clave para el público objetivo.
            5. Termina con un llamado a la acción claro.
            """
            
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt_generacion
                )
                
                st.success("✅ Prompt de Venta Generado:")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Ocurrió un error con la API de Gemini: {e}")