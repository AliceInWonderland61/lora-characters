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


def chat(message, persona, history):
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
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    history.append((message, answer))
    return history, ""


###########################################
#               AESTHETIC CSS             #
###########################################

custom_css = """
body {
    background: #FFF2EB !important;
    font-family: 'Inter', sans-serif;
    overflow-x: hidden;
}

/* Main Chat Container */
.gradio-container {
    background: #FFE8CD !important;
    border-radius: 25px;
    box-shadow: 0 0 30px rgba(255, 172, 135, 0.5);
}

/* Title */
#title {
    font-size: 38px;
    text-align: center;
    color: #D97B66;
    font-weight: 800;
    margin-bottom: 10px;
}

/* Chatbot */
.gr-chatbot {
    background: #FFF2EB !important;
    border-radius: 20px !important;
    border: 3px solid #FFD6BA !important;
    padding: 10px !important;
}

/* Input textbox */
textarea {
    background: #FFDCDC !important;
    border-radius: 18px !important;
    color: #4A2E2A !important;
    border: 2px solid #FFD6BA !important;
}

/* Buttons */
.gr-button {
    background: #FFD6BA !important;
    border-radius: 20px !important;
    color: #4A2E2A !important;
    font-weight: 700 !important;
    border: none !important;
    transition: 0.2s;
}

.gr-button:hover {
    background: #FFDEC6 !important;
    transform: scale(1.03);
}

/* Persona Radio Buttons */
.gr-radio input:checked + label {
    background: #FFDCDC !important;
    border-color: #D97B66 !important;
    border-radius: 12px !important;
}

/* Soft glow depending on persona */
.persona-jarvis {
    box-shadow: 0 0 25px rgba(200, 200, 255, 0.5);
}

.persona-sarcastic {
    box-shadow: 0 0 25px rgba(255, 80, 80, 0.5);
}

.persona-wizard {
    box-shadow: 0 0 25px rgba(120, 80, 255, 0.6);
}

/* Floating pumpkins + books */
.floating {
    position: fixed;
    font-size: 32px;
    animation: float 6s ease-in-out infinite;
    pointer-events: none;
    opacity: 0.9;
}

@keyframes float {
    0% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-20px) rotate(10deg); }
    100% { transform: translateY(0px) rotate(0deg); }
}

/* Falling leaves */
.leaf {
    position: fixed;
    top: -10vh;
    font-size: 24px;
    animation: fall linear infinite;
    opacity: 0.85;
    pointer-events: none;
}

@keyframes fall {
    0% { transform: translateY(-10vh) rotate(0deg); }
    100% { transform: translateY(110vh) rotate(360deg); }
}

/* Fairy sparkles */
.sparkle {
    position: fixed;
    width: 6px;
    height: 6px;
    background: white;
    border-radius: 50%;
    opacity: 0.8;
    box-shadow: 0 0 10px white;
    animation: sparkle 1.5s infinite ease-out;
}

@keyframes sparkle {
    from { transform: scale(0); opacity: 1; }
    to { transform: scale(1.5); opacity: 0; }
}
"""

###########################################
#     JAVASCRIPT FOR ANIMATION MAGIC ✨    #
###########################################

fancy_js = """
/* Falling leaves */
function spawnLeaf() {
    const leaf = document.createElement("div");
    leaf.classList.add("leaf");
    leaf.innerHTML = ["🍁","🍂","🍃"][Math.floor(Math.random()*3)];
    leaf.style.left = Math.random()*100 + "vw";
    leaf.style.fontSize = (20 + Math.random()*15) + "px";
    leaf.style.animationDuration = (4 + Math.random()*6) + "s";
    document.body.appendChild(leaf);
    setTimeout(()=>leaf.remove(), 12000);
}
setInterval(spawnLeaf, 700);


/* Floating pumpkins + books */
function spawnFloaters() {
    const float = document.createElement("div");
    float.classList.add("floating");
    float.innerHTML = ["📚","🎃"][Math.floor(Math.random()*2)];
    float.style.left = Math.random()*100 + "vw";
    float.style.top = (10 + Math.random()*70) + "vh";
    float.style.animationDuration = (4 + Math.random()*4) + "s";
    document.body.appendChild(float);
    setTimeout(()=>float.remove(), 8000);
}
setInterval(spawnFloaters, 4000);


/* Fairy sparkles */
function spawnSparkle() {
    const sp = document.createElement("div");
    sp.classList.add("sparkle");
    sp.style.left = Math.random()*100 + "vw";
    sp.style.top = Math.random()*100 + "vh";
    document.body.appendChild(sp);
    setTimeout(()=>sp.remove(), 1500);
}
setInterval(spawnSparkle, 500);


/* Persona glow */
function updatePersona(persona) {
    const chatbox = document.querySelector(".gr-chatbot");
    chatbox.classList.remove("persona-jarvis", "persona-sarcastic", "persona-wizard");
    chatbox.classList.add("persona-" + persona.toLowerCase());
}

window.updatePersona = updatePersona;
"""

###########################################
#      Build Pretty Interface ✨🎀         #
###########################################

with gr.Blocks(css=custom_css, js=fancy_js) as iface:
    gr.Markdown("<div id='title'>🍁 Cozy Fall Character Chat 🍂</div>")

    persona = gr.Radio(["Jarvis","Sarcastic","Wizard"], label="Choose Character", value="Jarvis")
    persona.change(fn=None, _js="updatePersona")

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Your Message 🌸")
    send = gr.Button("Send")

    send.click(chat, inputs=[msg, persona, chatbot], outputs=[chatbot, msg])

iface.launch()
