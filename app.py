from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = "./local_model"

app = FastAPI()

# Load model and tokenizer on startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
model.to(device)

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
def chat(req: ChatRequest):
    input_ids = tokenizer(req.prompt, return_tensors="pt").input_ids.to(device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"response": response[len(req.prompt):].strip()}
