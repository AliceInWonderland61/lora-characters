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

def chat_fn(history, message, persona):
    global current_adapter
    current_adapter = load_adapter(persona)

    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    outputs = current_adapter.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.8,
        do_sample=True
    )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    history.append({"role": "user", "content": message})
    history.append({"role": persona, "content": reply})
    return history, ""


# ==============================
#  FRONTEND + LEAVES + FONTS
# ==============================
HEADER_HTML = """
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500&family=Poppins:wght@400;600&family=Cormorant+Garamond:wght@500;700&display=swap" rel="stylesheet">

<div id="falling-leaves"></div>

<script>
// FALLING LEAVES
const leafImgs = [
    "/file/leaves/leaf1.png",
    "/file/leaves/leaf2.png",
    "/file/leaves/leaf3.png",
    "/file/leaves/leaf4.png"
];


function spawnLeaf() {
    const leaf = document.createElement("img");
    leaf.src = leafImgs[Math.floor(Math.random() * leafImgs.length)];
    leaf.classList.add("leaf");
    leaf.style.left = Math.random() * 100 + "vw";
    leaf.style.animationDuration = (6 + Math.random() * 6) + "s";
    leaf.style.width = (20 + Math.random() * 25) + "px";
    leafContainer.appendChild(leaf);

    setTimeout(() => leaf.remove(), 12000);
}
setInterval(spawnLeaf, 450);

// PERSONA FONT SWITCH (no Gradio JS hook)
document.addEventListener("click", () => {
    const selected = document.querySelector("input[type=radio][name=radio]:checked");
    if (!selected) return;
    const persona = selected.value;

    const box = document.querySelector("textarea");
    if (!box) return;

    box.classList.remove("jarvis-text","sarcastic-text","wizard-text");

    if (persona === "Jarvis") box.classList.add("jarvis-text");
    if (persona === "Sarcastic") box.classList.add("sarcastic-text");
    if (persona === "Wizard") box.classList.add("wizard-text");
});
</script>
"""

with gr.Blocks(css="custom.css") as ui:

    gr.HTML(HEADER_HTML)

    persona = gr.Radio(
        ["Jarvis", "Sarcastic", "Wizard"],
        label="Choose Character",
        value="Jarvis"
    )

    chatbot = gr.Chatbot(type="messages", height=350)
    msg = gr.Textbox(label="Your message", placeholder="Type here… 🍁")
    send_btn = gr.Button("Send")

    send_btn.click(
        chat_fn,
        inputs=[chatbot, msg, persona],
        outputs=[chatbot, msg]
    )

ui.launch()
