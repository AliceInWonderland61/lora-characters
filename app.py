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
# COMPREHENSIVE CSS THEME
# -----------------------------

custom_css = """
/* Autumn AI Character Chatbot - Custom Theme */

/* Main gradient background */
.gradio-container {
    background: linear-gradient(135deg, #D9A7C7, #FFFCDC) !important;
    font-family: 'Georgia', serif;
}

/* Main card styling */
.main-card {
    max-width: 1100px !important;
    margin: 20px auto !important;
    padding: 25px !important;
    background: rgba(255, 248, 240, 0.95) !important;
    border-radius: 22px !important;
    border: 3px solid #C88F6A !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.20) !important;
}

/* Character buttons */
.character-btn button {
    width: 100%;
    font-size: 18px !important;
    padding: 14px !important;
    border-radius: 14px !important;
    background: #FCE8D8 !important;
    border: 2px solid #C88F6A !important;
    color: #5C4033 !important;
}
.character-btn button:hover {
    background: #F7D9C4 !important;
}

/* AGGRESSIVE OVERRIDES FOR ALL GRAY/DARK BACKGROUNDS */

/* Target all possible container elements */
div, section, article, aside, nav, main,
.block, .panel, .form, .container,
[class*="Block"], [class*="block"],
[class*="Container"], [class*="container"],
[data-testid*="block"], [data-testid*="column"] {
    background-color: transparent !important;
}

/* Specifically target the gray rows */
.row, [class*="row"], [data-testid*="row"] {
    background: transparent !important;
    background-color: transparent !important;
}

/* Target columns */
.column, [class*="column"], [data-testid*="column"],
.gr-column, [class*="gr-column"] {
    background: transparent !important;
    background-color: transparent !important;
}

/* Character button row - force light background */
.gradio-row, [class*="gradio-row"] {
    background: rgba(252, 232, 216, 0.5) !important;
    padding: 10px !important;
    border-radius: 14px !important;
}

/* Input textbox and textarea */
input, textarea, select,
.input-text, .textbox,
[class*="input"], [class*="textbox"],
input[type="text"], textarea[class*="scroll"] {
    background: #FFF5E6 !important;
    background-color: #FFF5E6 !important;
    color: #5C4033 !important;
    border: 2px solid #E8D4C0 !important;
}

/* Chatbot container and messages */
.chatbot, [data-testid="chatbot"],
.chatbot *, [data-testid="chatbot"] *,
.message-wrap, .message, .message-row,
[class*="chatbot"], [class*="message"] {
    background: #1a1a1a !important;
    background-color: #1a1a1a !important;
    color: #E0E0E0 !important;
}

/* Individual chat messages */
.user-message, .bot-message,
[class*="user"], [class*="bot"],
.message.user, .message.bot {
    background: rgba(255, 245, 230, 0.1) !important;
    border-radius: 8px !important;
    padding: 8px !important;
}

/* Headers and labels */
h1, h2, h3, h4, h5, h6 {
    color: #5C4033 !important;
    background: transparent !important;
}

label, span, p {
    color: #5C4033 !important;
}

/* Button styling */
button {
    background: #E89B6C !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}

button:hover {
    background: #D88A5B !important;
}

/* Send button specific styling */
button[variant="primary"] {
    background: #E89B6C !important;
}

/* Audio component */
audio, [class*="audio"] {
    background: #FFF5E6 !important;
}

/* Checkbox */
input[type="checkbox"] {
    accent-color: #C88F6A !important;
}

/* Override any remaining dark/gray backgrounds with !important */
.dark, [class*="dark"],
.bg-gray, [class*="bg-gray"],
.bg-slate, [class*="bg-slate"] {
    background: #FFF5E6 !important;
    background-color: #FFF5E6 !important;
}

/* New Conversation button */
.clear-btn button, button:has(svg) {
    background: #95a5a6 !important;
    color: white !important;
}

.clear-btn button:hover {
    background: #7f8c8d !important;
}

/* Group elements (the boxes around sections) */
.gr-group, [class*="gr-group"],
.group, [class*="Group"] {
    background: rgba(255, 248, 240, 0.95) !important;
    border: 3px solid #C88F6A !important;
    border-radius: 22px !important;
    padding: 20px !important;
}
"""

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