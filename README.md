# 🏡 PropGenius: Retrieval-Augmented Generation (RAG) for Real Estate Copywriting
PropGenius is a Generative AI (GenAI) solution designed to optimize the creation of sales copy for the real estate sector. It leverages the RAG (Retrieval-Augmented Generation) architecture to anchor the language model to a Knowledge Base of descriptions from successfully sold properties, ensuring the generated copy is aligned with brand tone and proven market strategy.
This project demonstrates the end-to-end development of a scalable Machine Learning pipeline, from data ingestion to automated output testing.

## 🌟 Key Skills Demonstrated
| Area of Expertise | Project Component | Impact |
|---|---|---|
| Data Engineering / ETL | rag_core/data_pipeline.py Module | Simulation of historical data ingestion, chunking, and vectorization, preparing clean datasets for AI consumption. |
| ML Architecture (RAG) | rag_core/generator.py Module | Design and implementation of a complete RAG pipeline, decoupling external knowledge from the base LLM. |
| Testing and Data Quality | rag_core/evaluator.py Module | Implementation of LLM-Assisted Evaluation to measure output Faithfulness, preventing hallucinations and ensuring quality. |
| Cloud-Native Development | Streamlit App with Authentication | Interactive and user-friendly interface that manages application state and API Key injection for cloud-agnostic operation. |

## 🛠️ Technology Stack
* **Language:** Python
* **Interface/Dashboard:** Streamlit
* **LLM Models:** Gemini (Pro, Flash) via the google-genai library
* **Vector Database:** ChromaDB (Used locally to simulate Vertex AI Vector Search or AWS OpenSearch)
* **Frameworks:** LangChain (for document handling and retrieval).

## 🚀 System Architecture
The workflow follows a well-defined three-stage pipeline:
1. Data Ingestion (data_pipeline.py):
    * Historical sales data is loaded from data/historical_sales.txt.
    * Transformed into embeddings using the text-embedding-004 model.
    * Stored in the Vector Database (ChromaDB).
2. Augmented Generation (generator.py):
    * The user inputs the characteristics of the new property (the query).
    * The system retrieves the k most relevant historical documents via vector similarity search.
    * A Master Prompt is constructed that injects both the query and the retrieved context into the LLM (gemini-2.5-flash) to generate the final copy.
3. Quality Control (evaluator.py):
    * The generated copy is passed to an LLM "Judge" (gemini-2.5-pro).
    * The Judge evaluates the Faithfulness metric, verifying that factual information is derived only from the query or the context.
    * This acts as a system of continuous testing in production to monitor quality.

## 📋 How to Run the Project
### Prerequisites
* Python 3.10+
* A valid Google AI API Key.

### Clone the repository
```bash
git clone [https://github.com/tu-usuario/propgenius-rag.git](https://github.com/tu-usuario/propgenius-rag.git)
cd propgenius-rag
```

### Set up the virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Streamlit application:
```bash
streamlit run app.py
```


## Usage
The application will open in your browser.
1. Enter your Google AI API Key in the sidebar.
2. Once the key is validated, the RAG pipeline will initialize (data vectorization).
3. Enter the property characteristics and click "Generate Prompt and Evaluate".

---

### 🧑‍💻 Author
Joel Adid Rivas Mata

Senior Data Scientist | Data Engineer 

[LinkedIn](https://linkedin.com/in/joelrivasm)