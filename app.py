import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

LORA_ADAPTERS = {
    "Jarvis": "AlissenMoreno61/jarvis-lora",
    "Sarcastic": "AlissenMoreno61/sarcastic-lora",
    "Wizard": "AlissenMoreno61/wizard-lora",
}

# ---------- Cozy Fall CSS ----------
CUSTOM_CSS = """
body {
    background: linear-gradient(180deg, #f8ede3, #f9dcc4);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Main container */
.app-card {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    padding: 16px 20px;
    border: 1px solid #f3c8a1;
}

/* Title */
.app-title {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
    color: #7a4b2a;
    margin-bottom: 4px;
}

.app-subtitle {
    text-align: center;
    font-size: 0.9rem;
    color: #a06a3f;
    margin-bottom: 16px;
}

/* Persona radio buttons */
.persona-radio label {
    background: #ffe6c9 !important;
    border-radius: 999px !important;
    border: 1px solid #f0b98b !important;
    color: #7a4b2a !important;
    padding: 4px 10px !important;
}

.persona-radio input:checked + span {
    font-weight: 700 !important;
}

/* Chatbot */
.chatbox {
    height: 380px;
}

/* Message textbox */
.message-box textarea {
    border-radius: 999px !important;
    border: 1px solid #f0b98b !important;
}

/* Button */
.send-button {
    background: #d97b3d !important;
    color: white !important;
    border-radius: 999px !important;
    border: none !important;
}
.send-button:hover {
    background: #c0662b !important;
}

/* Falling leaves overlay */
.fall-container {
    pointer-events: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    z-index: 0;
    opacity: 0.7;
}

.leaf {
    position: absolute;
    top: -10%;
    font-size: 24px;
    animation: fall 12s linear infinite;
}

.leaf1 { left: 10%; animation-delay: 0s; }
.leaf2 { left: 30%; animation-delay: 3s; }
.leaf3 { left: 50%; animation-delay: 6s; }
.leaf4 { left: 70%; animation-delay: 1.5s; }
.leaf5 { left: 85%; animation-delay: 4.5s; }

@keyframes fall {
    0%   { transform: translateY(-10vh) translateX(0) rotate(0deg);   opacity: 0; }
    10%  { opacity: 1; }
    50%  { transform: translateY(50vh) translateX(-20px) rotate(90deg); }
    100% { transform: translateY(110vh) translateX(20px) rotate(180deg); opacity: 0; }
}
"""

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

current_persona = None
current_adapter = None


def switch_persona(persona: str):
    """Load the correct LoRA adapter if persona changed."""
    global current_persona, current_adapter

    if persona != current_persona or current_adapter is None:
        print(f"Switching persona → {persona}")
        adapter_repo = LORA_ADAPTERS[persona]
        current_adapter = PeftModel.from_pretrained(model, adapter_repo)
        current_adapter.eval()
        current_persona = persona


def generate_reply(message, history, persona):
    """Chat handler for the Gradio Chatbot."""
    if not message:
        return history

    switch_persona(persona)

    # Build a simple context from history + latest user message
    full_prompt = message
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    outputs = current_adapter.generate(
        **inputs,
        max_new_tokens=180,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    history = history + [(message, reply)]
    return history


with gr.Blocks(css=CUSTOM_CSS) as demo:
    # Falling leaves layer
    gr.HTML(
        """
        <div class="fall-container">
            <div class="leaf leaf1">🍁</div>
            <div class="leaf leaf2">🍂</div>
            <div class="leaf leaf3">🍁</div>
            <div class="leaf leaf4">🍂</div>
            <div class="leaf leaf5">🍁</div>
        </div>
        """
    )

    with gr.Column(elem_classes="app-card"):
        gr.HTML("<div class='app-title'>🍂 Cozy Fall Character Chat</div>")
        gr.HTML("<div class='app-subtitle'>Jarvis • Sarcastic • Wizard — one chat, three personalities.</div>")

        with gr.Row():
            with gr.Column(scale=3):
                persona = gr.Radio(
                    ["Jarvis", "Sarcastic", "Wizard"],
                    label="Choose Character",
                    value="Jarvis",
                    elem_classes="persona-radio",
                )
            with gr.Column(scale=9):
                chatbot = gr.Chatbot(label="Conversation", elem_classes="chatbox")
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your message",
                placeholder="Type your message here…",
                lines=2,
                elem_classes="message-box",
            )
            send = gr.Button("Send", elem_classes="send-button")

        send.click(
            generate_reply,
            inputs=[msg, chatbot, persona],
            outputs=chatbot,
        ).then(
            lambda: "",
            inputs=None,
            outputs=msg,
        )

demo.launch()
