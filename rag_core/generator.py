"""RAG/ML Pipeline"""


def generate_rag_prompt(client, vectorstore, prop_feat):
    """
    Executes the RAG pipeline. Prompt in Spanish because I need it that way :p
    hola mundo!
    """
    generator_model = 'gemini-2.5-flash'

    prop_str = (f"{prop_feat['type']} en {prop_feat['location']}. "
                f"Caracteristicas: {prop_feat['charact']}. "
                f"Público objetivo: {prop_feat['audience']}.")

    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    docs_retrieved = retriever.invoke(prop_str)

    context = "\n---\n".join([d.page_content for d in docs_retrieved])

    prompt = f"""
    Eres un experto en bienes raíces y copywriter con un tono {prop_feat['audience']}. específico.
    Genera una descripción de venta corta, convincente para una propiedad.

    **Información de la Propiedad:**
    - Características: {prop_feat['charact']}
    
    **CONTEXTO DE MARCA (EJEMPLOS EXITOSOS):**
    UTILIZA estos ejemplos de descripciones pasadas exitosas como REFERENCIA para replicar el tono:
    ---
    {context}
    ---

    **Requisitos de la Generación:** Título llamativo (máx. 10 palabras), 3-5 párrafos cortos, tono persuasivo.
    """

    response = client.models.generate_content(
        model=generator_model,
        contents=prompt
    )

    return response.text, context, prop_str
