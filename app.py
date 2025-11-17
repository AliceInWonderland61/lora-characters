"""
Autumn AI Character Chatbot (Two-Column Layout, Gradio 3 Compatible)
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
        "description": "Sophisticated AI Assistant",
        "personality": "Professional, articulate, British butler-like"
    },
    "Wizard": {
        "adapter": "AlissenMoreno61/wizard-lora",
        "emoji": "🍁",
        "description": "Mystical Sage of Autumn",
        "personality": "Poetic, uses medieval language, mystical"
    },
    "Sarcastic": {
        "adapter": "AlissenMoreno61/sarcastic-lora",
        "emoji": "🍃",
        "description": "Witty & Sharp-Tongued",
        "personality": "Wit, cheeky but helpful"
    }
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
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    history.append((message, response))

    audio = text_to_speech(response, character) if enable_tts else None
    return history, audio


# -----------------------------
# CSS (clean, centered)
# -----------------------------

custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #8B9DC3 0%, #C49A6C 30%, #DFB77B 60%, #E67E22 100%) !important;
    font-family: 'Georgia', serif;
}

.main-box, .content-box {
    max-width: 900px !important;
    padding: 25px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    border-radius: 20px !important;
    background: rgba(255, 248, 220, 0.94) !important;
    border: 3px solid #CD853F !important;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.25) !important;
}

footer { display: none !important; }

#character-radio label {
    text-align:center !important;
}
"""


# -----------------------------
# UI LAYOUT (TWO COLUMN)
# -----------------------------

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    # HEADER -----------------------------------------------------
    with gr.Group(elem_classes="main-box"):
        gr.HTML("""
            <h1 style='text-align:center;'>🍂 Autumn AI Characters 🍁</h1>
            <p style='text-align:center;'>Choose your cozy guide through the fall season</p>
        """)

    # HOW TO USE -------------------------------------------------
    with gr.Group(elem_classes="content-box"):
        gr.HTML("""
            <h3 style='text-align:center;'>🎯 How to Use Your Autumn AI</h3>
            <p style='text-align:center;'>
                🍂 Select a character below<br>
                🍁 Toggle voice if desired<br>
                🍃 Type your message<br>
                🎃 Enjoy the conversation!
            </p>
        """)

    # CHARACTER SELECTOR ----------------------------------------
    with gr.Group(elem_classes="content-box"):
        gr.HTML("<h3 style='text-align:center;'>🎭 Select Your Character</h3>")
        character_selector = gr.Radio(
            list(CHARACTERS.keys()),
            value="JARVIS",
            label="",
            elem_id="character-radio"
        )

    # MAIN TWO COLUMN LAYOUT -------------------------------------
    with gr.Row():

        # LEFT SIDE — Chat
        with gr.Column(scale=3):

            with gr.Group(elem_classes="content-box"):
                gr.HTML("<h3>💭 Conversation</h3>")
                chatbot = gr.Chatbot(height=360)

            with gr.Group(elem_classes="content-box"):
                gr.HTML("<h3>💬 Type Your Message</h3>")
                with gr.Row():
                    msg = gr.Textbox(lines=2, placeholder="Type here... 🍂", scale=5)
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

        # RIGHT SIDE — Character + Voice
        with gr.Column(scale=2):

            with gr.Group(elem_classes="content-box"):
                character_info = gr.HTML("")

            with gr.Group(elem_classes="content-box"):
                enable_tts = gr.Checkbox(label="🔊 Enable Voice", value=True)
                clear_btn = gr.Button("🔄 New Conversation")

            with gr.Group(elem_classes="content-box"):
                gr.HTML("<h3>🔊 Character Voice</h3>")
                audio_output = gr.Audio(type="filepath", autoplay=True)

    # FOOTER -----------------------------------------------------
    with gr.Group(elem_classes="main-box"):
        gr.HTML("<p style='text-align:center;'>🦊 Made with Gradio + LoRA</p>")

    # -------------------------
    # EVENT LOGIC
    # -------------------------

    def update_character_info(character):
        c = CHARACTERS[character]
        return f"""
            <h3 style='text-align:center;'>{c['emoji']} {character}</h3>
            <p style='text-align:center;'><strong>{c['description']}</strong></p>
            <p style='text-align:center;'>{c['personality']}</p>
        """

    character_selector.change(update_character_info, character_selector, character_info)

    submit_btn.click(
        chat_with_audio,
        [msg, chatbot, character_selector, enable_tts],
        [chatbot, audio_output]
    ).then(lambda: "", None, msg)

    msg.submit(
        chat_with_audio,
        [msg, chatbot, character_selector, enable_tts],
        [chatbot, audio_output]
    ).then(lambda: "", None, msg)

    clear_btn.click(lambda: ([], None), None, [chatbot, audio_output])


# -----------------------------
# LAUNCH
# -----------------------------
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

