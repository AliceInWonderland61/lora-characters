import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

LORA_ADAPTERS = {
    "Jarvis": "AlissenMoreno61/jarvis-lora",
    "Sarcastic": "AlissenMoreno61/sarcastic-lora",
    "Wizard": "AlissenMoreno61/wizard-lora"
}

print("Loading base model…")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def load_adapter(name):
    print(f"Loading adapter: {name}")
    return PeftModel.from_pretrained(model, LORA_ADAPTERS[name])

current_adapter = load_adapter("Jarvis")

def chat_fn(message, persona):
    global current_adapter
    current_adapter = load_adapter(persona)

    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    outputs = current_adapter.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.8,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ----------------------------------------------------------
# ⭐ FALL-STYLED HEADER: FONTS + LEAVES + COLORS
# ----------------------------------------------------------
HEADER_HTML = """
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500&family=Poppins:wght@400;600&family=Cormorant+Garamond:wght@500;700&display=swap" rel="stylesheet">

<style>
    body { 
        background: #FFF2EB !important;
    }

    .gradio-container {
        background: #FFF2EB !important;
    }

    /* Chatbox background */
    .gr-chatbot {
        background: #FFDCDC !important;
        border-radius: 10px;
        padding: 10px;
    }

    /* Input box */
    textarea {
        background: #FFE8CD !important;
        border-radius: 8px !important;
        border: 2px solid #FFD6BA !important;
    }

    /* Persona fonts */
    .jarvis-text { font-family: 'Playfair Display', serif !important; }
    .sarcastic-text { font-family: 'Poppins', sans-serif !important; }
    .wizard-text { font-family: 'Cormorant Garamond', serif !important; }

    /* Falling leaves container */
    #falling-leaves {
        pointer-events: none;
        position: fixed;
        top: 0; left: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        z-index: 0;
    }

    .leaf {
        position: absolute;
        width: 30px;
        opacity: 0.75;
        animation: fall linear infinite;
    }

    @keyframes fall {
        0% { transform: translateY(-10vh) rotate(0deg); }
        100% { transform: translateY(110vh) rotate(360deg); }
    }
</style>

<div id="falling-leaves"></div>

<script>
const leafContainer = document.getElementById("falling-leaves");
const leafImgs = [
    "file=leaves/leaf1.png",
    "file=leaves/leaf2.png",
    "file=leaves/leaf3.png",
    "file=leaves/leaf4.png"
];

function spawnLeaf() {
    const leaf = document.createElement("img");
    leaf.src = leafImgs[Math.floor(Math.random() * leafImgs.length)];
    leaf.classList.add("leaf");
    leaf.style.left = Math.random() * 100 + "vw";
    leaf.style.animationDuration = (6 + Math.random() * 7) + "s";
    leaf.style.opacity = 0.5 + Math.random() * 0.5;
    leaf.style.width = 20 + Math.random() * 30 + "px";
    leafContainer.appendChild(leaf);

    setTimeout(() => leaf.remove(), 14000);
}
setInterval(spawnLeaf, 500);

// Persona font updater
function updatePersona(p) {
    const box = document.querySelector("textarea");
    if (!box) return;

    box.classList.remove("jarvis-text","sarcastic-text","wizard-text");

    if (p === "Jarvis") box.classList.add("jarvis-text");
    if (p === "Sarcastic") box.classList.add("sarcastic-text");
    if (p === "Wizard") box.classList.add("wizard-text");
}
</script>
"""


# ----------------------------------------------------------
# ⭐ UI LAYOUT
# ----------------------------------------------------------
with gr.Blocks(css="body {background:#FFF2EB;}") as ui:

    gr.HTML(HEADER_HTML)

    persona = gr.Radio(
        ["Jarvis", "Sarcastic", "Wizard"],
        label="Choose Character",
        value="Jarvis"
    )

    # Fix: no _js needed here, use js() method
    persona.change(None, None, None).js("updatePersona")

    chatbox = gr.Chatbot(height=380)
    msg = gr.Textbox(label="Your message", placeholder="Type here… 🍁")
    send_btn = gr.Button("Send", variant="primary")

    send_btn.click(chat_fn, inputs=[msg, persona], outputs=chatbox)

ui.launch()
