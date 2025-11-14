import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------
# Load Base Model
# -----------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# -----------------------------
# Load LoRA Adapters (Your repos)
# -----------------------------
LORA_PATHS = {
    "Sarcastic": "AlissenMoreno61/sarcastic-lora",
    "Jarvis": "AlissenMoreno61/jarvis-lora",
    "Wizard": "AlissenMoreno61/wizard-lora"
}

adapters = {}
for name, repo in LORA_PATHS.items():
    adapters[name] = PeftModel.from_pretrained(base_model, repo)

current_character = "Sarcastic"


def set_character(character):
    global current_character
    current_character = character
    return f"🍁 Switched to **{character}** mode!"


def chat_fn(message, history):
    model = adapters[current_character]

    # Build conversation
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    # Chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )

    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = reply.split(message)[-1].strip()

    history.append((message, reply))
    return history, ""


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(css="custom.css", js="script.js") as demo:

    gr.HTML("""<h1 class="title">🍂 Fall Chat with Three Characters 🍁</h1>""")

    with gr.Row():
        gr.Button("😏 Sarcastic", elem_id="sarcastic").click(
            set_character, inputs=[], outputs=[]
        )
        gr.Button("🤖 Jarvis", elem_id="jarvis").click(
            set_character, inputs=[], outputs=[]
        )
        gr.Button("🧙 Wizard", elem_id="wizard").click(
            set_character, inputs=[], outputs=[]
        )

    chatbot = gr.Chatbot(height=450, elem_id="chatbox")
    msg = gr.Textbox(label="Send a message", placeholder="Type here...")

    msg.submit(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])

demo.launch()
