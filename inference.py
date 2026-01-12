import os
import torch

torch.set_num_threads(os.cpu_count())

from transformers import AutoTokenizer, AutoModelForCausalLM

model_path: str = os.getenv("MODEL_PATH")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )

device = torch.device("cpu")
model.to(device)

model.eval()

if hasattr(torch, 'compile'):
    model = torch.compile(model, mode="reduce-overhead")

system_prompt = """Você é um assistente especializado em chamadas de função.
Regras importantes:
- Chame **apenas** funções listadas
- Preencha corretamente todos os parâmetros obrigatórios
- Se a solicitação não corresponder a nenhuma função disponível, retorne: "Não especializado para esse tipo de ação"
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pagamento",
            "description": "Realiza um pagamento via Pix",
            "parameters": {
                "type": "object",
                "properties": {
                    "valor": {
                        "type": "number",
                        "description": "Valor do pagamento em reais (BRL)"
                    },
                    "forma_pagamento": {
                        "type": "string",
                        "description": "Forma de pagamento (pix, debito, credito)"
                    },
                    "imprimir": {
                        "type": "boolean",
                        "description": "Indica se deve imprimir o comprovante"
                    }
                },
                "required": ["valor", "forma_pagamento"]
            }
        }
    }
]

def run_inference(input: str, max_tokens: int = 64) -> str:

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input}
    ]

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
                    outputs[0][len(inputs["input_ids"][0]):],
                    skip_special_tokens=False
                )

        return generated


print(run_inference("30 reais débito"))

