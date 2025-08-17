import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import fitz
import os

# --- CONFIGURATION ---
BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "./qwen3-4b-hp-assistant-adapter" # Path to your trained LoRA adapter
PDF_PATH = "/users/student/pg/pg23/siddhant.gole/placement/chatbot/corpus/harrypotter.pdf"    
INDEX_PATH = "./faiss_hp_index"                # Where to save the vector database

# --- 1. LOAD THE FINE-TUNED MODEL ---

def load_model():
    """Loads the base model and merges the LoRA adapter."""
    print("Loading the fine-tuned model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    
    # Load and merge the LoRA adapter
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.merge_and_unload() # Merge weights and free up memory
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully.")
    return model, tokenizer

# --- 2. SETUP THE RAG KNOWLEDGE BASE ---

def create_or_load_rag_index(pdf_path, index_path):
    """Creates a FAISS vector database with improved parsing and chunking."""
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if os.path.exists(index_path):
        print(f"Loading existing FAISS index from {index_path}...")
        index = faiss.read_index(os.path.join(index_path, "hp.index"))
        with open(os.path.join(index_path, "chunks.txt"), "r", encoding="utf-8") as f:
            text_chunks = [line.strip() for line in f.readlines()]
        return index, text_chunks, embedding_model

    print(f"Creating new FAISS index from {pdf_path}...")
    
    # --- IMPROVEMENT 1: Use PyMuPDF for Better Text Extraction ---
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
            
    # Clean up the extracted text
    text = text.replace("\n", " ").strip()

    # --- IMPROVEMENT 2: Better Chunking Strategy ---
    # Split the text by paragraphs or large sections of text. A double space
    # can be a good heuristic for paragraph breaks after cleaning.
    raw_chunks = text.split("  ") # Split by double spaces, a common paragraph break
    
    # Further refine chunks: ensure they are not too short or too long
    MIN_CHUNK_LENGTH = 50 # Chunks must have at least 50 characters
    text_chunks = [chunk.strip() for chunk in raw_chunks if len(chunk.strip()) > MIN_CHUNK_LENGTH]

    if not text_chunks:
        raise ValueError("No text chunks were extracted from the PDF. Check the PDF quality and chunking strategy.")
        
    print(f"Extracted {len(text_chunks)} text chunks.")
    
    # Create embeddings
    embeddings = embedding_model.encode(text_chunks, convert_to_tensor=True, show_progress_bar=True).cpu().numpy()
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index and chunks for future use
    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, os.path.join(index_path, "hp.index"))
    with open(os.path.join(index_path, "chunks.txt"), "w", encoding="utf-8") as f:
        for chunk in text_chunks:
            f.write(chunk + "\n") # Write one chunk per line
        
    print("FAISS index created and saved.")
    return index, text_chunks, embedding_model

# --- 3. THE MAIN APPLICATION LOGIC ---

def answer_question(question, model, tokenizer, index, text_chunks, embedding_model):
    """Takes a question and generates an answer using RAG."""
    
    # Find relevant context from the book
    question_embedding = embedding_model.encode([question], convert_to_tensor=True).cpu().numpy()
    _, top_k_indices = index.search(question_embedding, k=3) # Retrieve top 3 chunks
    
    context = "\n---\n".join([text_chunks[i].strip() for i in top_k_indices[0]])
    
    # Create the prompt using the Qwen chat template
    messages = [
        {"role": "system", "content": "You are a knowledgeable and helpful guide to the Wizarding World. Answer the user's question based ONLY on the provided context from the official texts. Think step-by-step."},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate the answer
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # print("-" * 20)
    # print(f"❓ Question: {question}")
    # print(f"\n✅ Answer: {response}")
    # print("-" * 20)
    return response

def main():
    model, tokenizer = load_model()
    index, text_chunks, embedding_model = create_or_load_rag_index(PDF_PATH, INDEX_PATH)
    
    print("\n\n--- Harry Potter RAG Assistant ---")
    print("Your fine-tuned model is ready. Ask a question about the book! (Type 'quit' to exit)")
    
    while True:
        question = input("> ")
        if question.lower() == 'quit':
            break
        answer_question(question, model, tokenizer, index, text_chunks, embedding_model)

if __name__ == "__main__":
    main()