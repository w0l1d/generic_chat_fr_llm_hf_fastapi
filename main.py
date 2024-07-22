import torch
import transformers
import warnings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from fastapi.middleware.cors import CORSMiddleware

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Define the request and response models
class ChatRequest(BaseModel):
    user_id: str
    query: str


class ChatResponse(BaseModel):
    response: str
    conversation: List[Dict[str, str]]


# meta-llama/Meta-Llama-3-8B-Instruct is a large model that can be used for chatbot applications
# TinyLlama/TinyLlama-1.1B-Chat-v1.0 is a smaller model that can be used for chatbot applications
MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"


# Define the Llama3 class
class Llama3:
    def __init__(self, model_path, use_gpu=False):
        self.model_id = model_path
        model_kwargs = {
            "torch_dtype": torch.float16 if use_gpu else torch.float32,
            "low_cpu_mem_usage": True,
        }
        if use_gpu and torch.cuda.is_available():
            model_kwargs["quantization_config"] = {"load_in_4bit": True}

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs=model_kwargs,
        )
        self.tokenizer = self.pipeline.tokenizer
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(""),
        ]

    def get_response(self, query, message_history=None, max_tokens=4096, temperature=0.6,
                     top_p=0.9):
        if message_history is None:
            message_history = []
        user_prompt = message_history + [{"role": "user", "content": query}]
        prompt = self.tokenizer.apply_chat_template(
            user_prompt, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=self.terminators[0],
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(prompt):]
        return response, user_prompt + [{"role": "assistant", "content": response}]

    def chatbot(self, query, system_instructions="", initial_conversation=None):
        if initial_conversation is None:
            initial_conversation = [{"role": "system", "content": system_instructions}]
        conversation = initial_conversation
        response, updated_conversation = self.get_response(query, conversation)
        return response, updated_conversation


# Function to estimate how many models can fit into GPU memory
def get_max_models(model_class, model_path, use_gpu, model_memory_limit_gb):
    single_model_memory_gb = 8  # Assume each model approximately uses 8 GB, you may adjust this
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    available_memory_gb = total_memory_gb * 0.9  # Leave some buffer
    max_models = int(available_memory_gb // single_model_memory_gb)
    return max_models


if torch.cuda.is_available():
    max_models = get_max_models(Llama3, MODEL_PATH, use_gpu=True,
                                model_memory_limit_gb=15)
else:
    max_models = 1  # If no GPU is available, only load one model on CPU


# Initialize FastAPI and the thread pool executor
app = FastAPI()
executor = ThreadPoolExecutor(max_workers=max_models)  # Size of the model pool
models = Queue()
conversations = {}

# Initialize a pool of models
for _ in range(max_models):
    models.put(Llama3(MODEL_PATH, use_gpu=False))

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, you can restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the system instructions
SYSTEM_INSTRUCTIONS = (
    "Vous êtes un assistant utile dans un établissement d'enseignement. "
    "Vous aidez les futurs étudiants à comprendre les différents programmes offerts par l'établissement. "
    "Fournissez des réponses claires, concises et informatives à leurs questions concernant les cours, "
    "les compétences requises, les perspectives de carrière et d'autres informations connexes."
)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_id = request.user_id
    query = request.query

    if user_id not in conversations:
        conversations[user_id] = []

    # Acquire a model from the pool
    try:
        model = models.get(timeout=10)  # Wait up to 10 seconds for a model to become available
    except:
        raise HTTPException(status_code=503,
                            detail="All models are currently busy. Please try again later.")

    try:
        loop = asyncio.get_event_loop()
        response, updated_conversation = await loop.run_in_executor(
            executor, model.chatbot, query, SYSTEM_INSTRUCTIONS, conversations[user_id]
        )
        conversations[user_id] = updated_conversation
    finally:
        # Return the model to the pool
        models.put(model)

    return ChatResponse(response=response, conversation=updated_conversation)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
