🧙‍♂️ Pensieve AI: An Aligned Conversational RAG Agent
Pensieve AI is an end-to-end conversational agent designed to answer questions about the Harry Potter universe with high fidelity. This project demonstrates a complete MLOps workflow, including advanced model fine-tuning (SFT, DPO/RLHF), Retrieval-Augmented Generation (RAG) for contextual accuracy, and a human-in-the-loop data collection system deployed with a themed Streamlit UI.

✨ Features
Retrieval-Augmented Generation (RAG): The model's knowledge is grounded in the source text of the Harry Potter books, preventing hallucinations and ensuring answers are factually accurate.

Supervised Fine-Tuning (SFT): The base Qwen/Qwen3-4B-Instruct-2507 model is fine-tuned on a multi-domain dataset to enhance its conversational and reasoning abilities.

Preference Alignment (DPO/RLHF): The SFT model is further aligned to a specific "Wizarding World Expert" persona using Direct Preference Optimization (DPO) on a synthetically generated preference dataset.

Human-in-the-Loop Data Collection: The Streamlit app includes a feedback mechanism to continuously collect user preferences, allowing for iterative model improvement.

Interactive Themed UI: A custom-built Streamlit application provides an engaging, wizarding-themed user interface with real-time, streaming responses.

Efficient Inference: The final model is a 4-bit quantized PEFT adapter, ensuring low-latency responses on consumer-grade hardware.

### Interface

<img width="1446" height="872" alt="bot_1" src="https://github.com/user-attachments/assets/cedba8a9-7ab1-4fd0-9be1-d73cea69c468" />
<img width="1122" height="822" alt="bot-2" src="https://github.com/user-attachments/assets/202b2efb-32f9-4662-8162-4621199e6220" />
<img width="1087" height="887" alt="bot-3" src="https://github.com/user-attachments/assets/882ce452-ff59-41de-afe5-8bd67e733d03" />
<img width="981" height="890" alt="bot-4" src="https://github.com/user-attachments/assets/17da38e8-53e4-4e90-b824-52b66d127121" />


🛠️ Tech Stack
LLM: Qwen/Qwen3-4B-Instruct-2507

Fine-Tuning & Alignment: Hugging Face transformers, peft (QLoRA), trl (SFTTrainer, DPOTrainer)

RAG Pipeline:

Embeddings: sentence-transformers

Vector Store: faiss-cpu

Document Parsing: PyMuPDF

Deployment & UI: streamlit, streamlit-lottie

Core Libraries: PyTorch, accelerate, bitsandbytes

🏗️ Project Architecture
The project is divided into two main phases: an offline training pipeline and an online inference application.

Offline Training Pipeline:

SFT: A base LLM is fine-tuned on a general instruction dataset to improve its core capabilities.

DPO / RLHF: The SFT model is then aligned using a preference dataset, creating a specialized model that adheres to the desired persona and quality standards. This involves collecting preference pairs, training a reward model (for traditional RLHF), and finally, running the alignment optimization (DPO or PPO).

Online Inference & Data Collection (Streamlit App):

The final DPO-tuned adapter is loaded for inference.

A RAG pipeline retrieves relevant context from a pre-processed vector database of the Harry Potter books.

The model generates a response based on the user's query and the retrieved context.

The app periodically prompts the user for feedback on generated responses, saving their preferences to a log file to be used in the next iteration of the offline training pipeline.

🚀 Setup and Installation
Clone the repository:

Bash
cd repo-name
Create a Conda environment and install dependencies:

Bash

conda create -n wizard-ai python=3.10
conda activate wizard-ai
pip install torch torchvision torchaudio
pip install transformers peft trl bitsandbytes accelerate datasets
pip install faiss-cpu pypdf2 sentence-transformers streamlit streamlit-lottie requests PyMuPDF
pip install flash-attn --no-build-isolation
Place Data Files:

Place your Harry Potter PDF file in the root directory and name it harry_potter_book.pdf.

Place your DPO preference dataset in the root directory and name it preference_data.jsonl.

⚙️ Usage
The project is split into several scripts representing the different stages of the MLOps lifecycle.

1. Supervised Fine-Tuning (SFT)
This is the first training stage to improve the base model's general capabilities.

Bash

# (Assuming you have a train_sft.py script)
CUDA_VISIBLE_DEVICES=0 python train_sft.py
This will produce the qwen3-4b-hp-assistant-adapter.

2. Preference Tuning (DPO)
This stage aligns the SFT model with your preference data.

Bash

# This uses your SFT adapter and preference data
CUDA_VISIBLE_DEVICES=0 python train_dpo.py
This will produce the final qwen3-4b-hp-dpo-adapter.

3. Running the Interactive Application
This command launches the web app, which uses your final DPO-tuned adapter for inference and data collection.

Bash

streamlit run wizard_app.py
The app will open in your browser. The first time you run it, it will create the FAISS vector index from the PDF, which may take a few minutes. Subsequent launches will be much faster.
