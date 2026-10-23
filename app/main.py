import os
import time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI(
    title="CodeCraft-1.5B API",
    description="Containerized Local-First & Cloud-Native SLM for Enterprise Code Generation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./results/codecraft-1.5b-adapter")

tokenizer = None
model = None

@app.on_event("startup")
def load_model_and_tokenizer():
    global tokenizer, model
    print("Initializing inference runtime...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # Check if CUDA is available, otherwise load safely on CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Targeting device: {device}")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)

    if os.path.exists(ADAPTER_PATH):
        print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    else:
        print("Adapter not found. Falling back to base model.")
        model = base

    model.eval()
    print("Model loaded into memory successfully!")

class CodeGenerationRequest(BaseModel):
    prompt: str = Field(..., example="Write a Python function to find the reverse of a string")
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

@app.get("/health")
async def health():
    return {"status": "ok", "model": BASE_MODEL}

@app.post("/v1/generate")
async def generate_code(request: CodeGenerationRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is loading or unavailable.")

    start_time = time.time()

    formatted_prompt = (
        f"<|im_start|>system\nYou are CodeCraft-1.5B, an expert Python code synthesis assistant.<|im_end|>\n"
        f"<|im_start|>user\n{request.prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    total_tokens = output_ids.shape[1]
    new_tokens = total_tokens - prompt_tokens
    
    generated_text = tokenizer.decode(output_ids[0][prompt_tokens:], skip_special_tokens=True)
    
    # Clean up repetition loops or special tokens cleanly
    stop_phrases = ["You are CodeCraft", "<|im_end|>", "<|fim_pad|>"]
    for phrase in stop_phrases:
        if phrase in generated_text:
            generated_text = generated_text.split(phrase)[0]

    response_code = generated_text.strip()
    
    latency_ms = (time.time() - start_time) * 1000
    time_per_token_ms = latency_ms / max(1, new_tokens)

    return {
        "completion": response_code,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": new_tokens,
            "total_tokens": total_tokens
        },
        "performance": {
            "latency_ms": round(latency_ms, 2),
            "ms_per_token": round(time_per_token_ms, 2)
        }
    }