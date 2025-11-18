import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -----------------------------
#  Load Models
# -----------------------------
MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE)


# -----------------------------
#  Character Styles
# -----------------------------
CHAR_STYLES = {
    "JARVIS": "You are JARVIS, Tony Stark’s assistant. Respond professionally, intelligently, with formal tone.",
    "Wizard": "You are a wise wizard from a fantasy kingdom. Respond with magical, ancient, whimsical energy.",
    "Sarcastic": "You are sarcastic, dry, and witty. Respond with playful attitude."
}


# -----------------------------
#  Chat Function
# -----------------------------
def generate_response(character, message):
    if not message.strip():
        return ""

    prompt = CHAR_STYLES[character] + "\nUser: " + message + "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        inputs.input_ids,
        max_length=180,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in text:
        return text.split("Assistant:")[-1].strip()
    return text


# -----------------------------
#  Gradio UI
# -----------------------------
css = """
/* -------------------------
   Global Background
--------------------------*/
body {
    background: #EAF8E6 !important;  /* pastel mint */
}

/* -------------------------
   Main Card Wrapper
--------------------------*/
.main-card, .gr-panel, .gr-group {
    background: #FFF9EF !important; /* soft daisy beige */
    border-radius: 22px !important;
    padding: 20px !important;
    border: 3px solid #E4C89C !important; /* warm daisy brown */
}

/* -------------------------
   Chatbot Area (NO GRAY)
--------------------------*/
[data-testid="chatbot"] {
    background: #FFF9EF !important; /* beige */
    border-radius: 18px !important;
    border: 2px solid #F3DEC9 !important;
}

/* Chat messages */
.message.user {
    background: #FDE8D7 !important; /* pastel peach */
    border: 1px solid #F5D7C2 !important;
    color: #333 !important;
}

.message.bot {
    background: #FFFFFF !important; /* white */
    border: 1px solid #F2E6DA !important;
    color: #333 !important;
}

/* -------------------------
   Input Textbox
--------------------------*/
textarea, input[type="text"] {
    background: #FFFFFF !important;
    border: 2px solid #E4C89C !important;
    border-radius: 14px !important;
    color: #333 !important;
}

/* -------------------------
   Buttons
--------------------------*/
button {
    background: #FFE9A8 !important; /* pastel yellow */
    border: 2px solid #F1D48D !important;
    color: #5A4A2F !important;
    font-weight: 600 !important;
    border-radius: 14px !important;
}

button:hover {
    background: #FFEDB8 !important;
}

/* Character Buttons */
.character-btn {
    background: #FFE9A8 !important;
    border-radius: 14px !important;
    padding: 10px !important;
    border: 2px solid #F1D48D !important;
    font-weight: 600 !important;
}

/* -------------------------
   Voice Section
--------------------------*/
audio {
    background: #FFF9EF !important;
    border: 2px solid #F1D3BA !important;
    border-radius: 12px !important;
}
"""


# -----------------------------
#  UI Layout
# -----------------------------
def build_ui():

    with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="yellow", secondary_hue="green")) as demo:

        gr.Markdown(
            "<h1 style='text-align:center; color:#6B5C3D;'>🌼 Pastel Daisy Assistant 🌼</h1>"
        )

        with gr.Group(elem_classes="main-card"):
            with gr.Row():

                # Character Buttons
                with gr.Column(scale=1):
                    gr.Markdown(
                        "<h3 style='text-align:center; color:#6B5C3D;'>🌸 Choose Your Character</h3>"
                    )

                    character = gr.Radio(
                        choices=["JARVIS", "Wizard", "Sarcastic"],
                        value="JARVIS",
                        label="",
                        elem_classes="character-btn"
                    )

                    user_input = gr.Textbox(
                        placeholder="Type your message...",
                        label="",
                    )

                    send_btn = gr.Button("Send")

                # Chatbot
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Chatbot", height=420)

        # Voice Output Section
        with gr.Group(elem_classes="main-card"):
            enable_voice = gr.Checkbox(label="🎤 Enable Voice")
            audio_output = gr.Audio(label="Character Voice Output")

        new_chat_btn = gr.Button("🔄 New Conversation")

        # Logic
        def respond(character, message, chat_history):
            response = generate_response(character, message)
            chat_history.append((message, response))
            return "", chat_history
        
        send_btn.click(
            respond,
            [character, user_input, chatbot],
            [user_input, chatbot],
        )

        new_chat_btn.click(lambda: None, None, chatbot, queue=False)

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()
