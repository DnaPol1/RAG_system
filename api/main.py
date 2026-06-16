from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from configs.config import VEC_STORE_PATH, LOCAL_BAAI_PATH
from api.rag.pipeline import RAGPipline
from core.vectorStore import VectorStore

vector_store = VectorStore().load(VEC_STORE_PATH)
model = SentenceTransformer(LOCAL_BAAI_PATH)
pipline = RAGPipline(vector_store=vector_store, embedding_model=model)
app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
def get_page():
    with open(f"../frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask(request: QueryRequest):
    result = pipline.run(request.query)
    print(result)
    return {"answer": result["answer"]}