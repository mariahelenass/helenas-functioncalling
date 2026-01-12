import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

model_path: str = os.getenv("MODEL_PATH")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )

device = torch.device("cpu")
model.to(device)

model.eval()

system_prompt = "você é um assistente legal."

TOOLS = []

def run_inference(input: str, max_tokens: int = 128) -> str:

    messages = []

    inputs = tokenizer.apply_chat_template(
                messages,
                tools=TOOLS,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

    with torch.no_grad():
        outputs = model.generate(
                    **inputs.to(device),
                    max_new_tokens=max_tokens,
                    pad_token_id=tokenizer.eos_token_id
                )

        generated = tokenizer.decode(
                    outputs[0][len(inputs["inputs_ids"][0]):],
                    skip_special_tokens=False
                )

        return generated


print(run_inference("30 reais débito"))

