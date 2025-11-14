import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# CHANGE THIS to whichever adapter you want to load initially
LORA_ADAPTERS = {
    "Jarvis": "AlissenMoreno61/jarvis-lora",
    "Sarcastic": "AlissenMoreno61/sarcastic-lora",
    "Wizard": "AlissenMoreno61/wizard-lora"
}

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def load_adapter(adapter_repo):
    print(f"Loading adapter: {adapter_repo}")
    return PeftModel.from_pretrained(model, adapter_repo)

current_adapter = load_adapter(LORA_ADAPTERS["Jarvis"])

def chat(message, adapter_choice):
    global current_adapter

    # Reload adapter if user switched persona
    selected_repo = LORA_ADAPTERS[adapter_choice]
    current_adapter = load_adapter(selected_repo)

    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    outputs = current_adapter.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.8,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

with gr.Blocks() as iface:
    gr.Markdown("# 🧠 Multi-Persona LLaMA (Jarvis, Sarcastic, Wizard)")
    persona = gr.Dropdown(["Jarvis", "Sarcastic", "Wizard"], label="Persona")
    inp = gr.Textbox(label="Message")
    out = gr.Textbox(label="Response")
    btn = gr.Button("Send")
    btn.click(chat, inputs=[inp, persona], outputs=out)

iface.launch()
