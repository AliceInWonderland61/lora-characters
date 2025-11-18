"""
Autumn AI Character Chatbot — Light Autumn Air Theme 🍂✨
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from gtts import gTTS
import tempfile

# ----------------------------------
# MODEL + CHARACTERS
# ----------------------------------

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
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, CHARACTERS[character]["adapter"])
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
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True
    )

    history.append((message, response))

    audio = text_to_speech(response) if enable_tts else None
    return history, audio


# ----------------------------------
# LIGHT AUTUMN AIR CSS 🍂
# ----------------------------------

custom_css = """
/* ---------------------------
   LIGHT DAISY AESTHETIC 🌼✨
   Soft beige + thin daisy wallpaper
   --------------------------- */

.gradio-container {
    background: #E8D9B9 !important; /* warm beige */
    font-family: 'Georgia', serif;
    position: relative;
    overflow-x: hidden;
}

/* Daisy wallpaper layer */
.gradio-container::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 140%;
    height: 140%;
    pointer-events: none;

    background-image:
        radial-gradient(circle at 20% 20%, white 0 35%, transparent 36%),
        radial-gradient(circle at 80% 30%, white 0 40%, transparent 41%),
        radial-gradient(circle at 40% 80%, white 0 33%, transparent 34%),
        radial-gradient(circle at 70% 70%, white 0 38%, transparent 39%);

    /* Daisy centers */
    mask-image:
        radial-gradient(circle at 20% 20%, transparent 0 30%, black 31%),
        radial-gradient(circle at 80% 30%, transparent 0 35%, black 36%),
        radial-gradient(circle at 40% 80%, transparent 0 28%, black 29%),
        radial-gradient(circle at 70% 70%, transparent 0 33%, black 34%);

    opacity: 0.22;
    z-index: 0;

    /* soft movement feel */
    animation: drift 22s linear infinite;
}

@keyframes drift {
    0% { transform: translate(0px, 0px) scale(1.2); }
    50% { transform: translate(-35px, 20px) scale(1.25); }
    100% { transform: translate(0px, 0px) scale(1.2); }
}

/* Main Card Containers */
.main-card, .gr-group {
    position: relative;
    z-index: 2;
    max-width: 1100px !important;
    margin: 20px auto !important;
    padding: 25px !important;

    background: rgba(255, 255, 255, 0.88) !important;
    border-radius: 22px !important;
    border: 2px solid #DAC9A7 !important;
    box-shadow: 0 8px 18px rgba(0, 0, 0, 0.13) !important;
}

/* Character Buttons */
.character-btn button {
    width: 100%;
    font-size: 18px !important;
    padding: 14px !important;
    border-radius: 14px !important;

    background: #FFE9C7 !important;
    border: 2px solid #FFBF00 !important;
    color: #444 !important;
    font-weight: 600 !important;
}
.character-btn button:hover {
    background: #FFF4DE !important;
    transform: translateY(-2px);
    box-shadow: 0 3px 8px rgba(255, 191, 0, 0.25) !important;
}

/* Normal Buttons */
button {
    background: #FFBF00 !important;
    color: #333 !important;
    border-radius: 10px !important;
    border: 2px solid #E0A500 !important;
    font-weight: 600 !important;
}
button:hover {
    background: #FFD34D !important;
}

/* Text Inputs */
input, textarea {
    background: #FFFDF8 !important;
    border: 2px solid #E2C89F !important;
    color: #444 !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

/* Chatbot Bubble Styling */
.chatbot, [data-testid="chatbot"] {
    background: rgba(255, 255, 255, 0.85) !important;
    border-radius: 12px !important;
    border: 2px solid #E2C89F !important;
    color: #333 !important;
}

.message.user {
    background: #FFF1D8 !important;
    border-radius: 10px !important;
    color: #333 !important;
}
.message.bot {
    background: #FFFFFF !important;
    border-radius: 10px !important;
    color: #333 !important;
}

/* Headers */
h1, h2, h3, label, p, span {
    color: #333 !important;
}

/* Checkbox */
input[type="checkbox"] {
    accent-color: #FFBF00 !important;
}

/* Audio Box */
audio {
    background: white !important;
    border-radius: 12px !important;
    border: 2px solid #E2C89F !important;
}

"""


# ----------------------------------
# UI LAYOUT
# ----------------------------------

with gr.Blocks(css=custom_css) as demo:

    # MAIN CARD -------------------------------
    with gr.Group(elem_classes="main-card"):

        gr.HTML("<h2 style='text-align:center;'>🍂 Choose Your Character</h2>")

        # Character Buttons
        with gr.Row():
            char_buttons = []
            for name in CHARACTERS.keys():
                b = gr.Button(
                    f"{CHARACTERS[name]['emoji']} {name}",
                    elem_classes="character-btn"
                )
                char_buttons.append(b)

        character_state = gr.State("JARVIS")

        for btn, name in zip(char_buttons, CHARACTERS.keys()):
            btn.click(lambda x=name: x, outputs=character_state)

        # Chat columns (side by side)
        with gr.Row():

            # LEFT: Message input
            with gr.Column(scale=4):
                msg = gr.Textbox(
                    label="💬 Type your message",
                    lines=3,
                    placeholder="Say something..."
                )
                submit_btn = gr.Button("Send", variant="primary")

            # RIGHT: Chat
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(height=400)

    # AUDIO PANEL -----------------------------
    with gr.Group(elem_classes="main-card"):
        enable_tts = gr.Checkbox(label="🔊 Enable Voice", value=True)
        audio_output = gr.Audio(type="filepath", autoplay=True, label="Character Voice Output")

    # RESET -----------------------------------
    with gr.Group(elem_classes="main-card"):
        clear_btn = gr.Button("🔄 New Conversation")

    # Logic
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


# ----------------------------------
# LAUNCH
# ----------------------------------

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
