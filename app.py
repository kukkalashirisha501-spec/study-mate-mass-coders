# ============================================================
# 📘 StudyMate: AI-Powered PDF Q&A System (Hugging Face Version)
# ============================================================

!pip install -q gradio PyMuPDF sentence-transformers faiss-cpu transformers accelerate torch

import fitz  # PyMuPDF
import faiss
import torch
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ================== LOAD HUGGING FACE MODEL ==================
MODEL_NAME = "ibm-granite/granite-3.3-2b-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.3,
    top_p=0.9
)

# =================== PDF TEXT EXTRACTION =====================
def extract_text_from_pdfs(pdf_files):
    """Extract text content from uploaded PDF files"""
    all_text = ""
    for pdf_bytes in pdf_files:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")  # ✅ FIXED: pdf is bytes, not file object
        for page in doc:
            all_text += page.get_text("text") + "\n"
    return all_text.strip()

# ====================== TEXT CHUNKING ==========================
def chunk_text(text, chunk_size=500):
    """Split text into manageable chunks"""
    sentences = text.split(". ")
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

# =================== FAISS INDEX CREATION ======================
def build_faiss_index(chunks):
    """Embed chunks and create a FAISS index"""
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embedder, chunks

# ================== RETRIEVE RELEVANT CHUNKS ===================
def retrieve_chunks(query, embedder, index, chunks, top_k=3):
    """Find top-k relevant chunks for a query"""
    q_vec = embedder.encode([query])
    _, idxs = index.search(q_vec, top_k)
    return [chunks[i] for i in idxs[0]]

# ===================== GENERATE ANSWER =======================
def generate_answer(context, question):
    """Generate grounded answer using Hugging Face model"""
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer clearly and concisely using only the context above."
    response = qa_pipeline(prompt)
    return response[0]["generated_text"].replace(prompt, "").strip()

# ====================== GLOBAL STORAGE ========================
stored_index = None
stored_embedder = None
stored_chunks = None

# =================== PDF PROCESSING FUNCTION ===================
def process_pdfs(pdf_files):
    global stored_index, stored_embedder, stored_chunks
    if not pdf_files:
        return "⚠️ Please upload at least one PDF."

    text = extract_text_from_pdfs(pdf_files)
    if not text:
        return "⚠️ No text found in PDF."

    chunks = chunk_text(text)
    stored_index, stored_embedder, stored_chunks = build_faiss_index(chunks)
    return f"✅ PDFs processed successfully! {len(chunks)} text chunks created."

# ====================== Q&A FUNCTION ===========================
def ask_question(question):
    global stored_index, stored_embedder, stored_chunks
    if not all([stored_index, stored_embedder, stored_chunks]):
        return "⚠️ Please upload and process your PDFs first."
    if not question.strip():
        return "⚠️ Please enter a question."

    relevant = retrieve_chunks(question, stored_embedder, stored_index, stored_chunks)
    context = "\n".join(relevant)
    answer = generate_answer(context, question)
    return f"🧠 **Answer:**\n{answer}\n\n📚 **Context Used:**\n{context}"

# =========================== UI ===============================
with gr.Blocks(title="StudyMate - AI PDF Q&A") as demo:
    gr.Markdown("# 📘 StudyMate: AI-Powered PDF Q&A System")
    gr.Markdown("Upload your study PDFs and ask questions — answers are generated using IBM Granite 3.3-2B Instruct (Hugging Face).")

    with gr.Row():
        pdf_files = gr.File(label="📂 Upload one or more PDFs", file_count="multiple", type="binary")  # ✅ FIXED

    process_btn = gr.Button("🔍 Process PDFs")
    process_output = gr.Textbox(label="Processing Status")

    with gr.Row():
        question = gr.Textbox(label="💬 Ask your question here")

    ask_btn = gr.Button("🚀 Get Answer")
    answer_output = gr.Markdown()

    process_btn.click(fn=process_pdfs, inputs=[pdf_files], outputs=[process_output])
    ask_btn.click(fn=ask_question, inputs=[question], outputs=[answer_output])

demo.launch(share=True)
