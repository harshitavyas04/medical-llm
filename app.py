from flask import Flask, request, jsonify
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
import os

# Configuration
DATA_PATH = "NLP_FINAL.pdf"
DB_FAISS_PATH = "db_faiss"

app = Flask(__name__)

# Initialize model and vector store
def load_llm():
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.3
    )

def load_faiss_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    if os.path.exists(DB_FAISS_PATH):
        return FAISS.load_local(DB_FAISS_PATH, embeddings)
    else:
        raise ValueError("FAISS database not found. Please initialize it locally first.")

# Initialize QA system
llm = load_llm()
vector_store = load_faiss_vector_store()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True
)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    response = qa.invoke({"query": query})
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
