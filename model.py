import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.getenv("MODEL_PATH")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    local_files_only=True
)
model.to(device)
model.eval()



def inference(user_prompt: str):
    prompt = f"""<start_of_turn>user
    {user_prompt}
    <end_of_turn>
    <start_of_turn>model
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # removendo o prompt
    response = decoded.split("<start_of_turn>model")[-1].strip()

    return response
