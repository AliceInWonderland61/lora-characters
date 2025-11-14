import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

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

current_persona = None
current_adapter = None

def switch_persona(persona):
    global current_persona, current_adapter

    if persona != current_persona:
        print(f"Switching persona → {persona}")
        adapter_path = LORA_ADAPTERS[persona]
        current_adapter = PeftModel.from_pretrained(model, adapter_path)
        current_persona = persona

def chat(message, persona):
    switch_persona(persona)
    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    outputs = current_adapter.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

with gr.Blocks() as iface:
    gr.Markdown("# 🍁 Fall-Themed Multi-Persona Chatbot<br>Jarvis • Sarcastic • Wizard")

    persona = gr.Radio(["Jarvis", "Sarcastic", "Wizard"], label="Choose Persona")
    inp = gr.Textbox(label="Your Message")
    out = gr.Textbox(label="Bot Response", lines=7)
    btn = gr.Button("Send")

    btn.click(chat, inputs=[inp, persona], outputs=out)

iface.launch()
