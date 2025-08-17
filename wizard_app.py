import streamlit as st
from streamlit_lottie import st_lottie
import time
import requests
from typing import Generator
import json

# --- IMPORTANT ---
# Make sure your original 'app.py' file is in the same directory.
from app import load_model, create_or_load_rag_index

# --- Page Configuration ---
st.set_page_config(page_title="Wizarding World Assistant", page_icon="🧙‍♂️", layout="centered")

# --- Helper Functions ---

@st.cache_resource
def load_all_resources():
    """Loads all models and the RAG index once."""
    print("Loading all models and RAG index...")
    model, tokenizer = load_model()
    index, text_chunks, embedding_model = create_or_load_rag_index("./harry_potter_book.pdf", "./faiss_hp_index")
    return model, tokenizer, index, text_chunks, embedding_model

def get_ai_response(prompt: str, temperature: float = 0.7) -> str:
    """Gets a response from the RAG pipeline. Note: Temperature is a new parameter."""
    # This is a wrapper around your original answer_question logic
    # We need to recreate the logic here to accept the temperature parameter
    
    # Find relevant context
    question_embedding = embedding_model.encode([prompt], convert_to_tensor=True).cpu().numpy()
    _, top_k_indices = index.search(question_embedding, k=3)
    context = "\n---\n".join([text_chunks[i].strip() for i in top_k_indices[0]])
    
    # Create the prompt
    messages = [
        {"role": "system", "content": "You are a knowledgeable guide to the Wizarding World. Answer based ONLY on the provided context."},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{prompt}"}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    # Generate the answer with the specified temperature
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def save_feedback(prompt, chosen, rejected):
    """Saves the user's choice to a file."""
    with open("feedback_log.jsonl", "a") as f:
        f.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}) + "\n")
    st.toast("Thank you for your feedback, wizard!")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None

# --- App Layout & Logic ---

local_css("style.css")
model, tokenizer, index, text_chunks, embedding_model = load_all_resources()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you?"}]
if "response_count" not in st.session_state:
    st.session_state.response_count = 0
if "feedback_mode" not in st.session_state:
    st.session_state.feedback_mode = False
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

# Header
lottie_magic_wand = load_lottie_url("https://lottie.host/191b2398-59a9-4670-8910-c0892c55a5b1/u8aD0bO1mu.json")
if lottie_magic_wand:
    st_lottie(lottie_magic_wand, speed=1, height=150, key="wand")
st.title("The Wizarding World Assistant")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle feedback mode
if st.session_state.feedback_mode:
    st.info("Which response is better? Your feedback helps me learn!")
    
    with st.spinner("Conjuring two possible futures..."):
        # Generate two different answers by using different temperatures
        response_a = get_ai_response(st.session_state.last_prompt, temperature=0.7)
        response_b = get_ai_response(st.session_state.last_prompt, temperature=0.9)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(response_a)
        if st.button("Choose this one ✨", key="A"):
            save_feedback(st.session_state.last_prompt, response_a, response_b)
            st.session_state.messages.append({"role": "assistant", "content": response_a})
            st.session_state.feedback_mode = False
            st.rerun()

    with col2:
        st.markdown(response_b)
        if st.button("Choose this one 🪄", key="B"):
            save_feedback(st.session_state.last_prompt, response_b, response_a)
            st.session_state.messages.append({"role": "assistant", "content": response_b})
            st.session_state.feedback_mode = False
            st.rerun()

# Handle normal chat input
elif prompt := st.chat_input("Inscribe your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.last_prompt = prompt
    st.session_state.response_count += 1
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Casting a revealing spell..."):
            full_response = get_ai_response(prompt)
            st.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Check if it's time to ask for feedback after the 3rd response
    if st.session_state.response_count % 3 == 0:
        st.session_state.feedback_mode = True
        st.rerun()