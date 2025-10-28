"""Testing and quality evaluation for RAG systems."""

import json
from google.genai.errors import APIError
from pydantic import BaseModel

class EvaluationResult(BaseModel):
    """Schema for evaluation result."""
    faithfulness_score: int
    justification: str


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
    """

    try:
        llm_response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": EvaluationResult
            }
        )
        # print('LLM Response: ', llm_response.text)
        return json.loads(llm_response.text)
    except APIError as e:
        return {"faithfulness_score": None, "justification": f"Error del evaluador: {e}"}
