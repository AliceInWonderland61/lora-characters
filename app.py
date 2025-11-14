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

def load_adapter(name):
    return PeftModel.from_pretrained(model, LORA_ADAPTERS[name])

current_adapter = load_adapter("Jarvis")

def chat(message, persona):
    global current_adapter
    current_adapter = load_adapter(persona)

    prompt = f"You are {persona}. Stay strictly in character.\nUser: {message}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = current_adapter.generate(
        **inputs,
        max_new_tokens=180,
        temperature=0.8,
        top_p=0.95
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


#############################
#       CUSTOM  CSS         #
#############################

custom_css = """
body {
    background: #FFF2EB !important;
    font-family: 'Inter', sans-serif;
}

/* Chat container */
.gradio-container {
    background: #FFE8CD !important;
    border-radius: 20px;
}

/* Title */
#title {
    font-size: 32px;
    text-align: center;
    color: #D97B66;
    font-weight: 700;
    margin-bottom: 10px;
}

/* Input textbox */
.gr-textbox textarea {
    background: #FFDCDC !important;
    border-radius: 15px !important;
    color: #4A2E2A !important;
    border: 2px solid #FFD6BA !important;
}

/* Output box */
.gr-textbox-output {
    background: #FFF2EB !important;
    border-radius: 15px !important;
    padding: 15px;
    border: 2px solid #FFD6BA;
}

/* Buttons */
.gr-button {
    background: #FFD6BA !important;
    border-radius: 15px !important;
    color: #4A2E2A !important;
    border: none !important;
}

/* Persona buttons */
.gr-radio input:checked + label {
    background: #FFDCDC !important;
    border-color: #D97B66 !important;
}

/* Falling Leaves Animation */
@keyframes fall {
    0% { transform: translateY(-10vh) rotate(0deg); opacity: 1; }
    100% { transform: translateY(100vh) rotate(360deg); opacity: 0; }
}

.leaf {
    position: fixed;
    top: -10vh;
    font-size: 24px;
    animation: fall linear infinite;
    opacity: 0.8;
    pointer-events: none;
}
"""

#############################
# Falling leaves JS
#############################

falling_leaves_js = """
function createLeaf() {
    const leaf = document.createElement('div');
    leaf.classList.add('leaf');
    leaf.innerHTML = ['🍁','🍂','🍃'][Math.floor(Math.random()*3)];
    leaf.style.left = Math.random() * 100 + 'vw';
    leaf.style.animationDuration = (5 + Math.random() * 5) + 's';

    document.body.appendChild(leaf);

    setTimeout(() => leaf.remove(), 10000);
}

setInterval(createLeaf, 800);
"""

#############################
# Build interface
#############################

with gr.Blocks(css=custom_css, js=falling_leaves_js) as iface:
    gr.Markdown("<div id='title'>🍁 Cozy Fall Character Chat 🍂</div>")

    persona = gr.Radio(
        ["Jarvis", "Sarcastic", "Wizard"],
        label="Choose Character",
        value="Jarvis"
    )

    chatbot = gr.Chatbot(height=500)

    msg = gr.Textbox(label="Your Message")
    send = gr.Button("Send")

    send.click(chat, inputs=[msg, persona], outputs=chatbot)

iface.launch()
