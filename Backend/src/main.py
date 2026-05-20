from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from Backend.src.disease_identification import predict_disease
from Backend.src.fertilizer_recomendation import recommend_fertilizer
import argparse
import uvicorn
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
import os
import sys
from fastapi.responses import (
    FileResponse
)
from Backend.src.LLM import gemini_response
load_dotenv()
app = FastAPI(title="ARNA Backend")
PORT = int(os.environ.get("PORT", 8000))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# PLUG YOUR MODELS HERE
# ─────────────────────────────────────────────────────────────
def resourcePath(relativePath):
    try:
        basePath = sys._MEIPASS
    except Exception:
        basePath = os.path.abspath(".")
    return os.path.join(basePath, relativePath)



# disease_model = load_disease_model()
# rag_pipeline = load_rag_pipeline()
# llm = load_llm()

# ─────────────────────────────────────────────────────────────
# ENDPOINT 1: /analyze  (image → disease + fertilizer)
# ─────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Upload crop image → returns disease name + fertilizer name only.

    REPLACE the dummy response below with your actual model calls:
    
    image_bytes = await file.read()
    disease_result = disease_model.predict(image_bytes)
    fertilizer = rag_pipeline.retrieve(disease_result.label)
    return {
        "disease": disease_result.label,
        "confidence": disease_result.confidence,   # "High" / "Medium" / "Low"
        "fertilizer": fertilizer.name,
    }
    """
    image_bytes = await file.read()
   
    disease,confidence=predict_disease(image_bytes)
    fertilizer=recommend_fertilizer(disease)

    # await file.read()  # consume file — remove when using real model

    # ── DUMMY RESPONSE — replace with your model output ──────
    return {
        "disease": disease,          # string: disease label
        "confidence": confidence,               # string: High / Medium / Low
        "fertilizer": fertilizer,    # string: fertilizer name only
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINT 2: /chat  (message + history → reply)
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    # history: list = []     # [{"role": "user"/"assistant", "content": "..."}]
    # disease_context: str = ""   # pass detected disease so chat knows context

@app.post("/chat")
async def chat(body: ChatRequest):
    """
    RAG + LLM chat endpoint.

    REPLACE the dummy response below with your actual RAG + LLM pipeline:

    retrieved_docs = rag_pipeline.retrieve(body.message)
    context = "\\n".join(retrieved_docs)
    prompt = f"Disease context: {body.disease_context}\\nKnowledge: {context}\\nUser: {body.message}"
    reply = llm.generate(prompt, history=body.history)
    return {"reply": reply}
    """
    print(body.message)
    summary=gemini_response(body.message)
    # ── DUMMY RESPONSE — replace with your RAG + LLM output ──
    return {
        "reply": summary
    }


@app.get("/health")
def health():
    return {"status": "ok"}

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(resourcePath("Backend/src/frontend"), "static")),
    name="static",
)


@app.get("/{path_name:path}")
async def catch_all(path_name: str):
    return FileResponse(os.path.join(resourcePath("Backend/src/frontend"), "index.html"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML tool",
        epilog="Developed by CR Rao AIMSCS",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Hostname to run the server on (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )

    args = parser.parse_args()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=None,
        # reload=True,
    )