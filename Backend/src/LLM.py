from dotenv import load_dotenv
from Backend.src.Rag import FaissVectorStore, load_all_documents
import os
import google.generativeai as genai
import time
from google.api_core.exceptions import InternalServerError
# Set up Gemini AI


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "Backend/src/Dataset/Rag/faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gemma2-9b-it",
    ):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        load_dotenv()
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            # from data_loader import load_all_documents
            docs = load_all_documents("Backend/src/Dataset/Rag/data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        genai.configure(api_key=os.getenv("API_TOKEN"))
        self.llm = genai.GenerativeModel("gemma-4-26b-a4b-it")
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        # texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        texts = list(
            set(r["metadata"].get("text", "").strip() for r in results if r["metadata"])
        )

        # texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n".join(texts).strip()
        print("[context]:",context)
        if not context:
            return "I don't know based on the provided context."

        prompt = f"""
        
        Context:
        {context}

        Question:
        {query}

        
        Answer briefly using only the context.
        If answer is unavailable, say:
        I don't know based on the provided context."""
        response = None

        for attempt in range(3):
            try:
                response = self.llm.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": 50,
                        "temperature": 0.1
                    }
                )
                break

            except InternalServerError as e:
                print(f"Gemini API Error: {e}")
                time.sleep(5)

        if response is None:
            return "LLM service unavailable"
        return response


def gemini_response(text):
    rag_search = RAGSearch()
    # query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(text, top_k=3)
    print(summary)
    print("the text of the summar :",summary.text.strip())
    print("result")
    # print("[summary]: ", summary.text)
    # model_id = 'tunedModels/farmerqa-m3phv20xubea'

    # model = genai.GenerativeModel(model_id)

    # response = model.generate_content(text)
    return summary.text if summary else "❌ No response from Gemini AI."
