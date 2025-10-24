"""Module simulating an ETL pipeline for data vectorization."""

import os
from google import genai
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def load_and_vectorize_data(client, filepath='data/historical_sales.txt'):
    """
    Extracts, transforms and vectorize the historical data of successful sales.
    ETL Pipeline simulation
    """
    embedding_model = 'text-embedding-004'

    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    docs = [Document(page_content=d.strip()) for d in raw_text.split('\n') if d.strip()]
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, client=client)
    #embeddings = client.models.embed_content(model=embedding_model)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="successful_sales"
    )
    
    return vectorstore


if __name__ == '__main__':
    API_CLIENT = os.environ['GOOGLE_API_KEY']
    genai_client = genai.Client(api_key=API_CLIENT)
    load_and_vectorize_data(genai_client)
