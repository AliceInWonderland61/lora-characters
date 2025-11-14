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

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

current_adapter_name = None
current_adapter = None

def ensure_adapter(persona):
    global current_adapter, current_adapter_name
    repo = LORA_ADAPTERS[persona]
    if repo != current_adapter_name:
        current_adapter = PeftModel.from_pretrained(model, repo)
        current_adapter_name = repo

SYSTEM_PROMPTS = {
    "Jarvis": "You are Jarvis: polite, concise, helpful, professional, and calm.",
    "Sarcastic": "You are extremely sarcastic, witty, annoyed, and dramatic.",
    "Wizard": "You are a mystical fantasy wizard speaking in ancient magical tone."
}

def chat(message, persona):
    ensure_adapter(persona)

    prompt = (
        f"SYSTEM: {SYSTEM_PROMPTS[persona]}\n"
        f"USER: {message}\n"
        f"ASSISTANT:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = current_adapter.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # remove "SYSTEM:", "USER:" if model repeats
    for tag in ["SYSTEM:", "USER:", "ASSISTANT:"]:
        reply = reply.replace(tag, "")

    return reply.strip()


# ------------------- CSS --------------------
FALL_CSS = """
body {
    background: #F8E9DA; 
    font-family: 'Georgia', serif;
    color: #5A3E36;
}

/* Fall leaves animation */
@keyframes fall {
  0% { transform: translateY(-10vh) rotate(0deg); opacity: 1; }
  100% { transform: translateY(110vh) rotate(360deg); opacity: 0; }
}

.leaf {
    position: fixed;
    top: -10vh;
    font-size: 24px;
    pointer-events: none;
    animation: fall linear infinite;
}

/* Pink message bubbles */
.gr-chatbot-message {
    background: #FDECEF !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    color: #5A3E36 !important;
    border: 1px solid #E8B4B8 !important;
}

/* User bubbles */
.gr-chatbot-message.user {
    background: #FFECD5 !important;
}

/* Input box */
textarea, input {
    background: #FDECEF !important;
    border-radius: 12px !important;
    border: 1px solid #E8B4B8 !important;
}

/* Clean button */
button {
    background: #D8A47F !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 12px 20px !important;
    font-size: 16px !important;
}

"""

# ------------------- Leaves --------------------
import random
def falling_leaves_html():
    leaves = ""
    for i in range(16):
        emoji = random.choice(["🍁","🍂","🍃"])
        left = random.randint(0, 100)
        duration = random.uniform(6, 12)
        delay = random.uniform(0, 5)
        leaves += (
            f'<div class="leaf" style="left:{left}vw; '
            f'animation-duration:{duration}s; animation-delay:{delay}s;">{emoji}</div>'
        )
    return leaves


# ------------------- Interface --------------------
with gr.Blocks(css=FALL_CSS, head=falling_leaves_html()) as demo:
    
    gr.Markdown(
        """
        <h1 style='text-align:center; color:#A5633D;'>🍂 Cozy Fall Character Chat 🍁</h1>
        <p style='text-align:center;'>Soft autumn vibes • three personalities • one cozy chat</p>
        """
    )

    persona = gr.Radio(["Jarvis", "Sarcastic", "Wizard"], value="Jarvis", label="Choose Character")

    chatbot = gr.Chatbot(
        bubble_full_width=False,
        height=450,
        label=""
    )

    msg = gr.Textbox(placeholder="Type here… 🍁", label="Your message")
    send = gr.Button("Send")

    def respond(user_message, persona, history):
        bot_reply = chat(user_message, persona)
        history.append((user_message, bot_reply))
        return history, ""

    send.click(respond, [msg, persona, chatbot], [chatbot, msg])

demo.launch()
