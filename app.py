"""
Fall-Themed Character Chatbot
Hugging Face Space Application
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import random

# ============================================================================
# MODEL LOADING
# ============================================================================

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

# Load tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Character configurations
CHARACTERS = {
    "JARVIS": {
        "adapter": "AlissenMoreno61/jarvis-lora",
        "emoji": "🍂",
        "description": "Sophisticated AI Assistant",
        "color": "#4B8BBE"
    },
    "Wizard": {
        "adapter": "AlissenMoreno61/wizard-lora",
        "emoji": "🍁",
        "description": "Mystical Sage of Autumn",
        "color": "#9B59B6"
    },
    "Sarcastic": {
        "adapter": "AlissenMoreno61/sarcastic-lora",
        "emoji": "🍃",
        "description": "Witty & Sharp-Tongued",
        "color": "#E67E22"
    }
}

# Cache for loaded models
model_cache = {}

def load_character_model(character):
    """Load or retrieve cached character model"""
    if character not in model_cache:
        print(f"Loading {character}...")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(
            base_model,
            CHARACTERS[character]["adapter"]
        )
        model.eval()
        
        model_cache[character] = model
        print(f"✅ {character} loaded!")
    
    return model_cache[character]

# ============================================================================
# CHAT FUNCTION
# ============================================================================

def chat(message, history, character):
    """Generate response from selected character"""
    
    if not message.strip():
        return history
    
    # Load character model
    model = load_character_model(character)
    
    # Format chat history
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate response
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
    
    response = tokenizer.decode(
        outputs[0][len(inputs['input_ids'][0]):],
        skip_special_tokens=True
    )
    
    # Update history
    history.append((message, response))
    return history

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Custom CSS for fall theme
custom_css = """
#main-container {
    background: linear-gradient(135deg, #FFF8DC 0%, #FFE4B5 50%, #FFDAB9 100%);
}

.gradio-container {
    font-family: 'Arial', sans-serif;
}

#character-selector button {
    border-radius: 15px;
    padding: 15px;
    font-size: 16px;
    transition: all 0.3s ease;
}

#character-selector button:hover {
    transform: scale(1.05);
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}

.message {
    border-radius: 15px;
    padding: 10px 15px;
    margin: 5px;
}

#chatbot {
    border-radius: 20px;
    border: 3px solid #CD853F;
    overflow: hidden;
}

footer {
    display: none !important;
}
"""

# Build interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # 🍂 Autumn AI Characters 🍁
        ### Choose your guide through the fall season
        
        Experience three distinct AI personalities, each fine-tuned with LoRA adapters!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            character_selector = gr.Radio(
                choices=list(CHARACTERS.keys()),
                value="JARVIS",
                label="🎭 Select Character",
                elem_id="character-selector"
            )
            
            gr.Markdown("### Character Info")
            character_info = gr.Markdown(
                f"""
                **{CHARACTERS['JARVIS']['emoji']} JARVIS**
                
                {CHARACTERS['JARVIS']['description']}
                """
            )
            
            clear_btn = gr.Button("🔄 New Conversation", variant="secondary")
        
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
                elem_id="chatbot",
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    scale=4,
                    lines=1
                )
                submit_btn = gr.Button("Send 🚀", scale=1, variant="primary")
    
    gr.Markdown(
        """
        ---
        🍂 **Powered by LoRA Fine-tuning** 🍁 Built with Hugging Face Transformers 🍃
        
        Each character is a specialized adapter trained on unique personality data!
        """
    )
    
    # Update character info when selection changes
    def update_character_info(character):
        char_data = CHARACTERS[character]
        return f"""
        **{char_data['emoji']} {character}**
        
        {char_data['description']}
        """
    
    character_selector.change(
        fn=update_character_info,
        inputs=[character_selector],
        outputs=[character_info]
    )
    
    # Chat interactions
    msg.submit(
        fn=chat,
        inputs=[msg, chatbot, character_selector],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    submit_btn.click(
        fn=chat,
        inputs=[msg, chatbot, character_selector],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    clear_btn.click(lambda: [], outputs=[chatbot])

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860
    )