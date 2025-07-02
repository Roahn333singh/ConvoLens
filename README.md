# 🔍🧠 ConvoLens  


__**Conversation Insight Platform**__

ConvoLens is a cloud-native backend service designed to extract structured insights from long-form audio conversations such as debates, meetings, or interviews. 
It uses large language models (LLMs), vector search, and graph reasoning to transform unstructured dialogue into queryable, searchable knowledge.

---

__**✨ Features**__

- **Audio Transcription with Speaker Diarization**  
  Upload audio files with **diarization** to segment and identify individual speakers in a conversation.

- **Conversational Graph Generation**  
  Uses **LLM Graph Transformer** to build **knowledge graphs**, surfacing entities, relationships, and conversation structures.

- **Multimodal Search (Vector + Full-Text + Graph)**  
  Integrates vector embeddings, keyword-based, and graph-based search to deliver **deep insights** from unstructured conversations.

- **Purpose-Built for Analyzing Debates & Meetings**  
  Designed for navigating long-form discussions like **Sansad/Parliament debates**, empowering the public with searchable summaries.

- **LangChain & OpenAI Integration**  
  Leverages **LangChain experimental modules** and **OpenAI models** for smart text interpretation and extraction.

- **Cloud-Native Deployment**  
  Runs on **Google Cloud Run** using **Docker containers** for easy, scalable deployment.

---

__**🚀 API Endpoints**__

### `/api/transcribe`
- **Method:** `POST`  
- **Function:** Transcribes uploaded audio with speaker labels  
- **Input:** Audio file (e.g., WAV, MP3)  
- **Output:** JSON-formatted transcript with diarization

### `/generate-graph`
- **Method:** `POST`  
- **Function:** Converts text into a knowledge graph  
- **Input:** Text from a transcribed conversation  
- **Output:** Structured graph data (nodes & edges)

---

__**🛠 Tech Stack**__

- 🐍 **Python 3.10 (Slim Docker Image)**
- 🔥 **Flask** (API framework)
- 💡 **OpenAI LLMs**
- 🔗 **LangChain** (Core + Experimental modules)
- 🔧 **json-repair** (for correcting malformed JSON)
- 🧠 **Sentence Transformers + FAISS** (Vector search)
- 🌐 **Flask-CORS** (Cross-origin access)
- ☁️ **Google Cloud Build + Cloud Run**

---

__**📦 Deployment**__

```bash
# Step 1: Build and push Docker image to GCP Container Registry
gcloud builds submit --tag gcr.io/<your-project-id>/backend

# Step 2: Deploy to Cloud Run
gcloud run deploy flask-backend \
  --image gcr.io/<your-project-id>/backend \
  --platform managed \
  --region <your-region> \
  --allow-unauthenticated
<img width="1710" alt="image" src="https://github.com/user-attachments/assets/9708f577-6154-4798-9a39-dc16f4b7aa64" />

<img width="1613" alt="image" src="https://github.com/user-attachments/assets/76c74310-8dba-47d2-a59f-594acbf94182" />


