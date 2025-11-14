import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

LORA_ADAPTERS = {
    "Jarvis": "AlissenMoreno61/jarvis-lora",
    "Sarcastic": "AlissenMoreno61/sarcastic-lora",
    "Wizard": "AlissenMoreno61/wizard-lora"
}

# Load base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Load adapters only once
loaded_adapters = {}

def ensure_adapter_loaded(name):
    if name not in loaded_adapters:
        adapter_path = LORA_ADAPTERS[name]
        model.load_adapter(adapter_path, adapter_name=name)
        loaded_adapters[name] = True

    model.set_adapter(name)


def chat(message, persona):
    ensure_adapter_loaded(persona)
    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.8
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


custom_css = """
/* Soft pink textboxes */
.gradio-container .gr-textbox textarea {
    background-color: #ffd6e7 !important;
    border-radius: 10px !important;
    border: 1px solid #ffb8d2 !important;
    color: #4c2e33 !important;
    font-size: 16px !important;
    padding: 10px !important;
}

/* Dropdown styling */
.gradio-container .gr-dropdown select {
    background-color: #ffd6e7 !important;
    border-radius: 10px !important;
    border: 1px solid #ffb8d2 !important;
    color: #4c2e33 !important;
    font-size: 16px !important;
}

/* Button styling */
.gradio-container button {
    background-color: #ff8fb0 !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 18px !important;
    padding: 12px !important;
}
"""

with gr.Blocks(css=custom_css) as iface:
    gr.Markdown("<h1 style='text-align:center;'>🍁 Cozy Fall Character Chat</h1>")

    with gr.Column(scale=1):
        persona = gr.Dropdown(
            ["Jarvis", "Sarcastic", "Wizard"],
            label="Choose Character"
        )

        inp = gr.Textbox(
            label="Your Message",
            placeholder="Type your message here..."
        )

        out = gr.Textbox(
            label="Response",
            placeholder="The AI will reply here..."
        )

        btn = gr.Button("Send")
        btn.click(chat, inputs=[inp, persona], outputs=out)

iface.launch()
