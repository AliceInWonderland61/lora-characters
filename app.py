"""
Pastel Daisy Assistant
Pastel Forest + Pastel Sky Blue UI
Fully Cleaned + Correct CSS + No Purple Theme Override
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from gtts import gTTS
import tempfile


# ----------------------------
# MODEL SETUP
# ----------------------------
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

CHARACTERS = {
    "JARVIS": {
        "adapter": "AlissenMoreno61/jarvis-lora",
        "emoji": "🌼",
        "description": "Sophisticated AI Assistant",
        "personality": "Professional, articulate, British butler-like"
    },
    "Wizard": {
        "adapter": "AlissenMoreno61/wizard-lora",
        "emoji": "🪄",
        "description": "Mystical Forest Wizard",
        "personality": "Whimsical, magical, poetic"
    },
    "Sarcastic": {
        "adapter": "AlissenMoreno61/sarcastic-lora",
        "emoji": "🌿",
        "description": "Witty and Sharp",
        "personality": "Sarcastic but helpful"
    }
}

model_cache = {}

def load_character_model(character):
    if character not in model_cache:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(
            base_model, CHARACTERS[character]["adapter"]
        )
        model.eval()
        model_cache[character] = model
    return model_cache[character]


def text_to_speech(text, character):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except:
        return None


def chat_with_audio(message, history, character, enable_tts):
    if not message.strip():
        return history, None

    model = load_character_model(character)

    messages = []
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=150, temperature=0.7,
            top_p=0.9, repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        output[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    )

    history.append((message, response))
    audio = text_to_speech(response, character) if enable_tts else None
    return history, audio


# ----------------------------
# COLOR PALETTE
# ----------------------------
PASTEL_FOREST = "#A9C8A6"
PASTEL_BLUE = "#9DD1F5"
PASTEL_BLUE_HOVER = "#8CC7F0"
BLUE_BORDER = "#7BB8E0"
TEXT_DARK = "#4A4A4A"
CARD_WHITE = "#FFFFFF"


# ----------------------------
# CUSTOM CSS
# ----------------------------
custom_css = f"""
:root {{
    --color-accent: {PASTEL_BLUE} !important;
    --color-accent-soft: {PASTEL_BLUE} !important;
    --color-accent-subtle: {PASTEL_BLUE} !important;
    --button-primary-background: {PASTEL_BLUE} !important;
    --button-primary-text-color: #1E2A35 !important;
}}

.gradio-container {{
    background: {PASTEL_FOREST} !important;
    font-family: 'Quicksand', sans-serif;
}}

/* Main card */
.main-card {{
    max-width: 1250px;
    margin: 0 auto;
    padding: 22px;
    border-radius: 22px;
    background: {CARD_WHITE};
    border: 3px solid {BLUE_BORDER};
    box-shadow: 0 6px 18px rgba(0,0,0,0.10);
}}

.section-card {{
    background: {CARD_WHITE};
    border: 3px solid {BLUE_BORDER};
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.10);
    margin-bottom: 22px;
}}

h1, h3 {{
    color: {TEXT_DARK} !important;
}}

button, .gr-button {{
    background: {PASTEL_BLUE} !important;
    color: #1E2A35 !important;
    border-radius: 14px !important;
    border: 2px solid {BLUE_BORDER} !important;
    font-weight: 600 !important;
    transition: 0.15s ease-in-out;
}}

button:hover, .gr-button:hover {{
    background: {PASTEL_BLUE_HOVER} !important;
}}

.character-btn {{
    padding: 10px 5px;
    border-radius: 14px;
    background: {PASTEL_BLUE};
    border: 2px solid {BLUE_BORDER};
    text-align: center;
    cursor: pointer;
    font-weight: 600;
    color: #1E2A35;
}}

.character-btn:hover {{
    background: {PASTEL_BLUE_HOVER};
}}

#chatbot {{
    border-radius: 18px !important;
    border: 2px solid {BLUE_BORDER} !important;
    background: #ffffff !important;
}}

input, textarea {{
    border-radius: 12px !important;
    border: 2px solid {BLUE_BORDER} !important;
}}
"""


# ----------------------------
# UI LAYOUT
# ----------------------------
with gr.Blocks(css=custom_css) as demo:

    gr.HTML("<h1 style='text-align:center;'>🌼 Pastel Daisy Assistant 🌼</h1>")

    with gr.Column(elem_classes="main-card"):

        # Character buttons
        gr.HTML("<h2 style='text-align:center; font-weight:800; font-size:28px;'>Choose Your Character</h2>")
        char_btns = gr.Radio(
            list(CHARACTERS.keys()),
            value="JARVIS",
            label="",
            elem_classes="character-btn"
        )

        # Chat layout
        with gr.Row():
            with gr.Column(scale=1, elem_classes="section-card"):
                msg = gr.Textbox(placeholder="Type your message…", lines=3)
                send_btn = gr.Button("Send")

            with gr.Column(scale=2, elem_classes="section-card"):
                chatbot = gr.Chatbot(height=350, elem_id="chatbot")

        # Voice section
        with gr.Column(elem_classes="section-card"):
            enable_voice = gr.Checkbox(label="🔊 Enable Voice Output", value=True)
            audio_output = gr.Audio(type="filepath", autoplay=True, label="Character Voice")

        new_conv_btn = gr.Button("New Conversation")

    # Logic
    send_btn.click(chat_with_audio,
        [msg, chatbot, char_btns, enable_voice],
        [chatbot, audio_output]
    ).then(lambda: "", outputs=msg)

    msg.submit(chat_with_audio,
        [msg, chatbot, char_btns, enable_voice],
        [chatbot, audio_output]
    ).then(lambda: "", outputs=msg)

    new_conv_btn.click(lambda: ([], None), outputs=[chatbot, audio_output])


if __name__ == "__main__":
    demo.launch()