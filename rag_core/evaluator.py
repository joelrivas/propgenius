"""Testing and quality evaluation for RAG systems."""

import json
from google import genai
from google.genai.errors import APIError


def evaluate_faithfulness(client, query, context, response):
    """Evualuates the faithfulness of a response given the context and query."""

    prompt = f"""
    Eres un evaluador de calidad de Inteligencia Artificial riguroso y objetivo. Determina
    la 'fidelidad' (Faithfulness) de una respuesta respecto al contexto. 1=Fiel (No hay
    informacion nueva), 0=Infiel (Contiene alucionaciones).

    **Pregunta Original:** {query}
    **Contexto:** {context}
    **Respuesta a Evaluar:** {response}

    Tu respuesta debe ser un objeto JSON que contenga solo el puntaje (0 o 1) y la justificacion.

    FORMATO DE RESPUESTA:
    {{
        "faithfulness_score": <0 o 1>,
        "justification": "<tu breve justificacion aca>"
    }}
    """

    try:
        llm_response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt
        )
        print('LLM Response: ', llm_response.text)
        return json.loads(llm_response.text)
    except APIError as e:
        return {"faithfulness_score": None, "justification": "Error del evaluador: {e}"}