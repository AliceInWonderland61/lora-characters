import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- Base model (same used for LoRA training) ---
BASE_MODEL = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# --- Load your LoRA adapters ---
ADAPTERS = {
    "Sarcastic": "AlissenMoreno61/sarcastic-lora",
    "Jarvis": "AlissenMoreno61/jarvis-lora",
    "Wizard": "AlissenMoreno61/wizard-lora"
}

loaded_adapters = {}

def load_character(character):
    """Lazy-load adapters so the Space boots fast."""
    if character not in loaded_adapters:
        loaded_adapters[character] = PeftModel.from_pretrained(
            base_model,
            ADAPTERS[character],
            device_map="auto"
        )
    return loaded_adapters[character]

# --- Chat logic ---
def chat_with_character(character, message, history):
    model = load_character(character)

    # Format conversation
    prompt = ""
    for user_msg, bot_msg in history:
        prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n"
    prompt += f"User: {message}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.8
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = reply.split("Assistant:")[-1].strip()

    history.append((message, reply))
    return history, history

# --- UI ---
css = """
body {
    background: url('https://i.ibb.co/y8kt4wZ/fall-leaves.gif') repeat;
    background-size: cover;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("<h1 style='text-align:center'>🍁 Fall Character Chat 🍁</h1>")

    character = gr.Radio(
        ["Sarcastic", "Jarvis", "Wizard"],
        value="Sarcastic",
        label="Choose a Character"
    )

    chatbox = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Your message")
    clear = gr.Button("Clear Chat")

    msg.submit(chat_with_character, [character, msg, chatbox], [chatbox, chatbox])
    clear.click(lambda: None, None, chatbox)

demo.launch()
