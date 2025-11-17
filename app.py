"""
Fall-Themed Character Chatbot with Text-to-Speech
Beautiful Autumn Design with Falling Leaves Animation
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from gtts import gTTS
import tempfile

# ============================================================================
# MODEL LOADING
# ============================================================================

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

CHARACTERS = {
    "JARVIS": {
        "adapter": "AlissenMoreno61/jarvis-lora",
        "emoji": "🍂",
        "description": "Sophisticated AI Assistant",
        "gradient": "linear-gradient(135deg, #4B8BBE, #5A9FD4)",
        "voice_speed": 0.9,
        "voice_lang": "en"
    },
    "Wizard": {
        "adapter": "AlissenMoreno61/wizard-lora",
        "emoji": "🍁",
        "description": "Mystical Sage of Autumn",
        "gradient": "linear-gradient(135deg, #9B59B6, #8E44AD)",
        "voice_speed": 0.8,
        "voice_lang": "en"
    },
    "Sarcastic": {
        "adapter": "AlissenMoreno61/sarcastic-lora",
        "emoji": "🍃",
        "description": "Witty & Sharp-Tongued",
        "gradient": "linear-gradient(135deg, #E67E22, #D35400)",
        "voice_speed": 1.1,
        "voice_lang": "en"
    }
}

model_cache = {}

def load_character_model(character):
    if character not in model_cache:
        print(f"Loading {character}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, CHARACTERS[character]["adapter"])
        model.eval()
        model_cache[character] = model
        print(f"✅ {character} loaded!")
    return model_cache[character]

# ============================================================================
# TEXT TO SPEECH
# ============================================================================

def text_to_speech(text, character):
    """Convert character's response to speech"""
    try:
        char_config = CHARACTERS[character]
        tts = gTTS(
            text=text,
            lang=char_config["voice_lang"],
            slow=(char_config["voice_speed"] < 1.0)
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

# ============================================================================
# CHAT FUNCTION
# ============================================================================

def chat_with_audio(message, history, character, enable_tts):
    if not message.strip():
        return history, None
    
    model = load_character_model(character)
    
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
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
    
    history.append((message, response))
    
    audio_file = None
    if enable_tts:
        audio_file = text_to_speech(response, character)
    
    return history, audio_file

# ============================================================================
# GRADIO INTERFACE - FALL THEME
# ============================================================================

# Beautiful Fall-Themed CSS
custom_css = """
/* Main Container - Fall Background */
.gradio-container {
    background: linear-gradient(135deg, #8B9DC3 0%, #DFB77B 50%, #E67E22 100%) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Falling Leaves Animation */
@keyframes fall {
    0% {
        transform: translateY(-10vh) rotate(0deg);
        opacity: 1;
    }
    100% {
        transform: translateY(110vh) rotate(360deg);
        opacity: 0.7;
    }
}

@keyframes sway {
    0%, 100% {
        transform: translateX(0);
    }
    50% {
        transform: translateX(30px);
    }
}

.leaf {
    position: fixed;
    top: -10vh;
    z-index: 1;
    pointer-events: none;
    animation: fall linear infinite, sway ease-in-out infinite;
    font-size: 2rem;
    filter: drop-shadow(2px 2px 3px rgba(0,0,0,0.3));
}

/* Character Selection Buttons */
.character-btn {
    border-radius: 20px !important;
    padding: 20px !important;
    margin: 10px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
    border: 3px solid transparent !important;
    background: rgba(255, 255, 255, 0.9) !important;
}

.character-btn:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3) !important;
    border-color: #D35400 !important;
}

/* Chatbot Container */
#chatbot {
    border-radius: 25px !important;
    border: 4px solid #8B4513 !important;
    background: rgba(255, 248, 240, 0.95) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
}

/* Message Bubbles */
.message.user {
    background: linear-gradient(135deg, #E67E22, #D35400) !important;
    color: white !important;
    border-radius: 20px 20px 5px 20px !important;
    padding: 12px 18px !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2) !important;
}

.message.bot {
    background: rgba(255, 255, 255, 0.95) !important;
    border: 2px solid #DFB77B !important;
    border-radius: 20px 20px 20px 5px !important;
    padding: 12px 18px !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
}

/* Input Box */
.input-box {
    border: 3px solid #8B4513 !important;
    border-radius: 15px !important;
    padding: 15px !important;
    background: rgba(255, 248, 240, 0.9) !important;
    font-size: 16px !important;
}

/* Send Button */
.send-btn {
    background: linear-gradient(135deg, #E67E22, #D35400) !important;
    color: white !important;
    border-radius: 15px !important;
    padding: 15px 30px !important;
    font-weight: bold !important;
    border: none !important;
    transition: all 0.3s ease !important;
}

.send-btn:hover {
    background: linear-gradient(135deg, #D35400, #C0392B) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3) !important;
}

/* Audio Player */
.audio-player {
    border: 3px solid #8B4513 !important;
    border-radius: 15px !important;
    background: rgba(255, 248, 240, 0.9) !important;
    padding: 15px !important;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2) !important;
}

/* Header Styling */
h1 {
    color: #5D4037 !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3) !important;
    font-weight: bold !important;
}

h3 {
    color: #6D4C41 !important;
}

/* Info Box */
.info-box {
    background: rgba(255, 248, 240, 0.9) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    border: 3px solid #8B4513 !important;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2) !important;
}

/* Checkbox Styling */
.checkbox-label {
    font-size: 16px !important;
    color: #5D4037 !important;
    font-weight: 600 !important;
}
"""

# JavaScript for Falling Leaves Animation
falling_leaves_js = """
<script>
function createFallingLeaves() {
    const leaves = ['🍂', '🍁', '🍃'];
    const container = document.body;
    
    function createLeaf() {
        const leaf = document.createElement('div');
        leaf.className = 'leaf';
        leaf.innerHTML = leaves[Math.floor(Math.random() * leaves.length)];
        leaf.style.left = Math.random() * 100 + 'vw';
        leaf.style.animationDuration = (Math.random() * 10 + 10) + 's';
        leaf.style.animationDelay = Math.random() * 5 + 's';
        container.appendChild(leaf);
        
        setTimeout(() => {
            leaf.remove();
        }, 20000);
    }
    
    // Create initial leaves
    for(let i = 0; i < 15; i++) {
        setTimeout(createLeaf, i * 300);
    }
    
    // Continuously create new leaves
    setInterval(createLeaf, 2000);
}

// Start animation when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createFallingLeaves);
} else {
    createFallingLeaves();
}
</script>
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), head=falling_leaves_js) as demo:
    
    gr.HTML("""
        <div style='text-align: center; padding: 30px; background: rgba(139, 69, 19, 0.1); border-radius: 20px; margin-bottom: 20px;'>
            <h1 style='font-size: 3.5em; margin: 0; color: #5D4037;'>🍂 Autumn AI Characters 🍁</h1>
            <p style='font-size: 1.3em; color: #6D4C41; margin-top: 10px;'>✨ Choose your guide through the fall season ✨</p>
            <p style='font-size: 1.1em; color: #8B4513;'>Experience three distinct personalities with voice responses!</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.HTML("<div style='text-align: center; padding: 15px; background: rgba(255, 248, 240, 0.9); border-radius: 15px; border: 3px solid #8B4513;'><h2 style='color: #5D4037; margin: 0;'>🎭 Select Character</h2></div>")
            
            character_selector = gr.Radio(
                choices=list(CHARACTERS.keys()),
                value="JARVIS",
                label="",
                elem_classes=["character-btn"]
            )
            
            enable_tts = gr.Checkbox(
                label="🔊 Enable Character Voice",
                value=True,
                info="Hear your character speak aloud!",
                elem_classes=["checkbox-label"]
            )
            
            gr.HTML("<div class='info-box' style='margin-top: 20px;'><h3 style='margin-top: 0;'>📋 Character Info</h3></div>")
            
            character_info = gr.Markdown(
                f"""
                **{CHARACTERS['JARVIS']['emoji']} JARVIS**
                
                {CHARACTERS['JARVIS']['description']}
                
                🎤 Voice: Professional & Measured
                """,
                elem_classes=["info-box"]
            )
            
            clear_btn = gr.Button("🔄 New Conversation", variant="secondary", size="lg")
        
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="💬 Conversation",
                height=450,
                elem_id="chatbot",
                bubble_full_width=False,
                avatar_images=(
                    "https://api.dicebear.com/7.x/avataaars/svg?seed=user",
                    "https://api.dicebear.com/7.x/bottts/svg?seed=ai"
                )
            )
            
            audio_output = gr.Audio(
                label="🔊 Character Voice Response",
                type="filepath",
                autoplay=True,
                elem_classes=["audio-player"]
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="",
                    placeholder="Type your message here... 🍂",
                    scale=4,
                    lines=1,
                    elem_classes=["input-box"]
                )
                submit_btn = gr.Button("Send 🚀", scale=1, variant="primary", elem_classes=["send-btn"])
    
    gr.HTML("""
        <div style='text-align: center; padding: 25px; background: rgba(139, 69, 19, 0.1); border-radius: 20px; margin-top: 30px;'>
            <h3 style='color: #5D4037; margin-bottom: 15px;'>🎯 How to Use</h3>
            <p style='color: #6D4C41; font-size: 1.1em;'>
                <strong>1.</strong> Choose your character 🎭 &nbsp;&nbsp;
                <strong>2.</strong> Enable voice responses 🔊 &nbsp;&nbsp;
                <strong>3.</strong> Start chatting! 💬
            </p>
            <p style='color: #8B4513; margin-top: 15px; font-size: 0.9em;'>
                🍂 Powered by LoRA Fine-tuning • 🍁 Built with Gradio • 🍃 Voices by gTTS
            </p>
            <p style='color: #A0522D; margin-top: 10px; font-size: 0.85em;'>
                Created by <strong>AlissenMoreno61</strong>
            </p>
        </div>
    """)
    
    # Update character info
    def update_character_info(character):
        char_data = CHARACTERS[character]
        voice_desc = {
            "JARVIS": "Professional & Measured (0.9x speed)",
            "Wizard": "Deep & Mysterious (0.8x speed)",
            "Sarcastic": "Quick & Energetic (1.1x speed)"
        }
        return f"""
        **{char_data['emoji']} {character}**
        
        {char_data['description']}
        
        🎤 Voice: {voice_desc[character]}
        """
    
    character_selector.change(
        fn=update_character_info,
        inputs=[character_selector],
        outputs=[character_info]
    )
    
    # Chat interactions
    msg.submit(
        fn=chat_with_audio,
        inputs=[msg, chatbot, character_selector, enable_tts],
        outputs=[chatbot, audio_output]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    submit_btn.click(
        fn=chat_with_audio,
        inputs=[msg, chatbot, character_selector, enable_tts],
        outputs=[chatbot, audio_output]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    clear_btn.click(
        lambda: ([], None),
        outputs=[chatbot, audio_output]
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860
    )