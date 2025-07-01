from flask import Flask, request, jsonify
from utils.llm import OpenRouterLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from openai import OpenAI
import requests
import time
import json
from flask_cors import CORS

import os

app = Flask(__name__)
# CORS(app)  # 👈 add this after initializing Flask

CORS(app, resources={r"/*": {"origins": "*"}})
# --- AssemblyAI Settings ---
ASSEMBLYAI_KEY = "Replace with your env-secured key"  # Replace with your env-secured key
HEADERS = {"authorization": ASSEMBLYAI_KEY}
TRANSCRIBE_HEADERS = {**HEADERS, "content-type": "application/json"}

# In-memory session
cache = {}
@app.route("/")
def home():
    return "✅ Flask backend is up and running!"
# --- Upload Audio ---
@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    file = request.files['file']
    res = requests.post("https://api.assemblyai.com/v2/upload", headers=HEADERS, data=file.read())
    res.raise_for_status()
    return jsonify({"upload_url": res.json()["upload_url"]})

UPLOAD_ENDPOINT = "https://api.assemblyai.com/v2/upload"
TRANSCRIBE_ENDPOINT = "https://api.assemblyai.com/v2/transcript"

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    # Step 1: Upload file to AssemblyAI
    upload_response = requests.post(
        UPLOAD_ENDPOINT,
        headers=HEADERS,
        files={'file': file}
    )
    upload_response.raise_for_status()
    audio_url = upload_response.json()['upload_url']

    # Step 2: Start transcription
    transcribe_response = requests.post(
        TRANSCRIBE_ENDPOINT,
        headers=HEADERS,
        json={"audio_url": audio_url, "speaker_labels": True}
    )
    transcribe_response.raise_for_status()
    transcript_id = transcribe_response.json()["id"]
    polling_url = f"{TRANSCRIBE_ENDPOINT}/{transcript_id}"

    # Step 3: Poll for transcription result
    while True:
        status_response = requests.get(polling_url, headers=HEADERS)
        status_data = status_response.json()
        if status_data["status"] == "completed":
            break
        elif status_data["status"] == "failed":
            return jsonify({"error": "Transcription failed"}), 500
        time.sleep(5)

    utterances = status_data.get("utterances", [])
    
    # Format output as requested
    content = [
        {
            "actor": f"SPEAKER_{str(u['speaker']).zfill(2)}",
            "dialogue": u['text']
        }
        for u in utterances
    ]

        # 🧠 Rebuild transcript string for schema compatibility
    transcript_string = "\n".join([f"{c['actor']}: {c['dialogue']}" for c in content])

    # ✅ Return both: transcript string (required by consumer) + your detailed structure
    return jsonify({
        "transcript": transcript_string,
        "root": {
            "content": content
        }
    })




@app.route('/generate-graph', methods=['POST'])
def generate_graph():
    # 🔁 Clear cache before rebuilding
    cache.clear()
    data = request.json
    transcript = data.get("transcript")

    # If transcript not directly provided, try to extract from "root.content"
    if not transcript:
        try:
            content = data["root"]["content"]
            transcript = "\n".join([f"{entry['actor']}: {entry['dialogue']}" for entry in content])
        except Exception:
            return jsonify({"error": "Transcript not provided"}), 400

    doc = Document(page_content=transcript)
    llm = OpenRouterLLM(api_key='Replace with Your own openrouterllm key')
    transformer = LLMGraphTransformer(llm=llm)
    graph_docs = transformer.convert_to_graph_documents([doc])

    # Embedding for QA
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.create_documents([transcript])
    embedding = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en")
    vector_store = FAISS.from_documents(split_docs, embedding)

    # Cache for use in /query
    cache["graph_docs"] = graph_docs
    cache["vector_store"] = vector_store
    cache["transcript"] = transcript

    # Return simple graph metadata
    nodes = [{"id": n.id, "type": n.type, "detail": getattr(n, "detail", "")} for n in graph_docs[0].nodes]
    rels = [{"source": r.source.id, "target": r.target.id, "type": r.type} for r in graph_docs[0].relationships]

    return jsonify({"nodes": nodes, "relationships": rels})



# --- Query & Summarize ---
@app.route('/query', methods=['POST'])
def query():
    query_text = request.json.get("query")
    if not query_text:
        return jsonify({"error": "Query not provided"}), 400

    vector_store = cache.get("vector_store")
    transcript = cache.get("transcript", "")
    graph_docs = cache.get("graph_docs", [])

    if not vector_store:
        return jsonify({"error": "Vector store not found"}), 500

    vector_results = vector_store.similarity_search(query_text, k=3)
    fulltext_results = [line for line in transcript.splitlines() if any(word in line.lower() for word in query_text.lower().split())]
    
    matched_nodes = [n for n in graph_docs[0].nodes if any(word in n.id.lower() for word in query_text.lower().split())]
    matched_rels = [r for r in graph_docs[0].relationships if any(word in r.type.lower() for word in query_text.lower().split())]

    formatted_vector = "\n".join([f"- {doc.page_content}" for doc in vector_results])
    formatted_fulltext = "\n".join([f"- {line}" for line in fulltext_results])
    formatted_nodes = "\n".join([f"- {n.id} ({n.type})" for n in matched_nodes])
    formatted_rels = "\n".join([f"- {r.source.id} --[{r.type}]--> {r.target.id}" for r in matched_rels])

    # prompt = f"""You are a helpful assistant. Based on the following data, answer the question concisely and try to highlight key points don't print * pattern.\n\nQuestion: {query_text}\n\n📊 Vector Matches:\n{formatted_vector or 'No results.'}\n\n📄 Fulltext Matches:\n{formatted_fulltext or 'No results.'}\n\n🧠 Knowledge Graph Matches:\nNodes:\n{formatted_nodes or 'No nodes.'}\nRelations:\n{formatted_rels or 'No relations.'}"""
    prompt = f"""
    You are a helpful assistant.

    Using the data below, write a concise and relevant answer to the user query.

    ## Guidelines:
    - Do not use markdown syntax (no asterisks, no double asterisks).
    - Do not write in ALL CAPS.
    - Do not overuse numbered lists — use short paragraphs unless the structure demands it.
    - Emphasize key terms by isolating them in the sentence (e.g., 'The missile used was a **cluster munition**.' becomes 'The missile used was a cluster munition, designed to maximize casualties.').
    - Do not guess. Only summarize what is supported by the data.
    - Use a clear tone. No flowery or dramatic language.
    - Font style such as 'Resist Mono' will be applied by the frontend, not here.
    - Use bullet Points where required.

    QUESTION:
    {query_text}

    📊 VECTOR MATCHES:
    {formatted_vector or 'No results.'}

    📄 FULLTEXT MATCHES:
    {formatted_fulltext or 'No results.'}

    🧠 KNOWLEDGE GRAPH MATCHES:
    NODES:
    {formatted_nodes or 'No nodes.'}
    RELATIONS:
    {formatted_rels or 'No relations.'}
    """


    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key='Replace with Your own openrouterllm key')
    response = client.chat.completions.create(model="deepseek/deepseek-chat-v3-0324:free", messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ])
    return jsonify({"answer": response.choices[0].message.content.strip()})

@app.route('/debug/cache', methods=['GET'])
def view_cache():
    try:
        # Use safe serialization for display
        def safe_serialize(obj):
            if isinstance(obj, str):
                return obj
            if isinstance(obj, list):
                return [safe_serialize(i) for i in obj]
            if hasattr(obj, '__dict__'):
                return str(obj)
            return str(obj)

        serialized_cache = {
            key: safe_serialize(value)
            for key, value in cache.items()
        }

        return jsonify({"cache": serialized_cache})
    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)

