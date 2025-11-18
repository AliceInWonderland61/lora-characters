"""
Autumn AI Character Chatbot — Clean Button Layout (Gradio 3 Compatible)
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from gtts import gTTS
import tempfile

# -----------------------------
# MODEL + CHARACTER CONFIG
# -----------------------------

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

CHARACTERS = {
    "JARVIS": {
        "adapter": "AlissenMoreno61/jarvis-lora",
        "emoji": "🍂",
    },
    "Wizard": {
        "adapter": "AlissenMoreno61/wizard-lora",
        "emoji": "🍁",
    },
    "Sarcastic": {
        "adapter": "AlissenMoreno61/sarcastic-lora",
        "emoji": "🍃",
    },
}

model_cache = {}

def load_character_model(character):
    if character not in model_cache:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, CHARACTERS[character]["adapter"])
        model.eval()
        model_cache[character] = model
    return model_cache[character]

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except:
        return None

def chat_fn(message, history, character, enable_tts):
    if not message.strip():
        return history, None

    model = load_character_model(character)

    messages = []
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    history.append((message, response))

    audio = text_to_speech(response) if enable_tts else None
    return history, audio

# -----------------------------
# LOAD EXTERNAL CSS
# -----------------------------

# Read the CSS from external file
with open("autumn_theme.css", "r") as f:
    custom_css = f.read()

# -----------------------------
# UI LAYOUT (buttons + two columns)
# -----------------------------

with gr.Blocks(css=custom_css) as demo:

    # MAIN CARD ---------------------------------------------------
    with gr.Group(elem_classes="main-card"):

        gr.HTML("<h2 style='text-align:center;'>🍂 Choose Your Character</h2>")

        # CHARACTER BUTTONS ROW
        with gr.Row():
            char_buttons = []
            for c in CHARACTERS.keys():
                btn = gr.Button(f"{CHARACTERS[c]['emoji']} {c}", elem_classes="character-btn")
                char_buttons.append(btn)

        # Hidden variable to store selected character
        character_state = gr.State("JARVIS")

        # Link buttons to character state
        for btn, name in zip(char_buttons, CHARACTERS.keys()):
            btn.click(lambda x=name: x, outputs=character_state)

        # TWO-COLUMN CHAT AREA
        with gr.Row():

            # LEFT — message input
            with gr.Column(scale=4):
                msg = gr.Textbox(label="💬 Type your message", lines=3)
                submit_btn = gr.Button("Send", variant="primary")

            # RIGHT — conversation
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(height=400)

    # AUDIO BOX ---------------------------------------------------
    with gr.Group(elem_classes="main-card"):
        enable_tts = gr.Checkbox(label="🔊 Enable Voice", value=True)
        audio_output = gr.Audio(type="filepath", autoplay=True, label="Character Voice Output")

    # RESET BUTTON ------------------------------------------------
    with gr.Group(elem_classes="main-card"):
        clear_btn = gr.Button("🔄 New Conversation")

    # -------------------------
    # Event Logic
    # -------------------------

    submit_btn.click(
        chat_fn,
        inputs=[msg, chatbot, character_state, enable_tts],
        outputs=[chatbot, audio_output],
    ).then(lambda: "", outputs=msg)

    msg.submit(
        chat_fn,
        inputs=[msg, chatbot, character_state, enable_tts],
        outputs=[chatbot, audio_output],
    ).then(lambda: "", outputs=msg)

    clear_btn.click(lambda: ([], None), outputs=[chatbot, audio_output])

# -----------------------------
# RUN
# -----------------------------

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)