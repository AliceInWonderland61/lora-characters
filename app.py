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
        "personality": "Professional, articulate, British butler-like",
        "voice_speed": 0.9,
        "voice_lang": "en"
    },
    "Wizard": {
        "adapter": "AlissenMoreno61/wizard-lora",
        "emoji": "🍁",
        "description": "Mystical Sage of Autumn",
        "personality": "Poetic, uses medieval language, mystical",
        "voice_speed": 0.8,
        "voice_lang": "en"
    },
    "Sarcastic": {
        "adapter": "AlissenMoreno61/sarcastic-lora",
        "emoji": "🍃",
        "description": "Witty & Sharp-Tongued",
        "personality": "Ryan Reynolds wit, cheeky but helpful",
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
# GRADIO INTERFACE - REFINED FALL THEME
# ============================================================================

custom_css = """
/* Main Container - Warm Fall Gradient */
.gradio-container {
    background: linear-gradient(135deg, #8B9DC3 0%, #C49A6C 30%, #DFB77B 60%, #E67E22 100%) !important;
    font-family: 'Georgia', 'Times New Roman', serif;
}

/* Falling Leaves Animation */
@keyframes fall {
    0% {
        transform: translateY(-10vh) rotate(0deg);
        opacity: 1;
    }
    100% {
        transform: translateY(110vh) rotate(360deg);
        opacity: 0.5;
    }
}

@keyframes sway {
    0%, 100% { transform: translateX(0); }
    50% { transform: translateX(30px); }
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

/* Remove default Gradio styling */
.container { gap: 0 !important; }
footer { display: none !important; }

/* Character Selection Radio Buttons */
#character-radio label {
    background: rgba(255, 248, 220, 0.95) !important;
    border: 3px solid #8B4513 !important;
    border-radius: 15px !important;
    padding: 15px 20px !important;
    margin: 8px 0 !important;
    font-size: 18px !important;
    font-weight: bold !important;
    color: #5D4037 !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}

#character-radio label:hover {
    background: rgba(255, 235, 205, 1) !important;
    transform: translateX(5px) !important;
    box-shadow: 0 5px 15px rgba(139, 69, 19, 0.3) !important;
}

#character-radio input:checked + label {
    background: linear-gradient(135deg, #DFB77B, #CD853F) !important;
    color: white !important;
    border-color: #8B4513 !important;
    box-shadow: 0 5px 20px rgba(139, 69, 19, 0.5) !important;
}

/* Chatbot Styling */
#chatbot {
    border-radius: 20px !important;
    border: 4px solid #8B4513 !important;
    background: rgba(255, 248, 240, 0.98) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
}

.message.user {
    background: linear-gradient(135deg, #CD853F, #8B4513) !important;
    color: white !important;
}

.message.bot {
    background: rgba(255, 248, 220, 0.95) !important;
    border: 2px solid #DFB77B !important;
}

/* Text Input */
.input-box textarea {
    border: 3px solid #8B4513 !important;
    border-radius: 15px !important;
    background: rgba(255, 248, 240, 0.98) !important;
    font-size: 16px !important;
    color: #5D4037 !important;
}

.input-box textarea:focus {
    border-color: #CD853F !important;
    box-shadow: 0 0 10px rgba(205, 133, 63, 0.5) !important;
}

/* Send Button */
button.primary {
    background: linear-gradient(135deg, #CD853F, #8B4513) !important;
    color: white !important;
    border: none !important;
    border-radius: 15px !important;
    padding: 15px 30px !important;
    font-weight: bold !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #B8860B, #654321) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(139, 69, 19, 0.4) !important;
}

/* Secondary Button (Clear) */
button.secondary {
    background: rgba(139, 69, 19, 0.1) !important;
    color: #8B4513 !important;
    border: 2px solid #8B4513 !important;
    border-radius: 15px !important;
}

button.secondary:hover {
    background: rgba(139, 69, 19, 0.2) !important;
}

/* Checkbox */
.checkboxgroup label, .checkbox label {
    color: #5D4037 !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

input[type="checkbox"] {
    accent-color: #8B4513 !important;
}

/* Info Cards */
.info-card {
    background: rgba(255, 248, 220, 0.95) !important;
    border: 3px solid #8B4513 !important;
    border-radius: 15px !important;
    padding: 20px !important;
    margin: 10px 0 !important;
    box-shadow: 0 5px 15px rgba(139, 69, 19, 0.2) !important;
}

/* Audio Player */
audio {
    border: 3px solid #8B4513 !important;
    border-radius: 15px !important;
    background: rgba(255, 248, 220, 0.95) !important;
}

/* Headers */
h1, h2, h3 {
    color: #5D4037 !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2) !important;
}

/* Remove unnecessary borders and spacing */
.block { border: none !important; }
.padded { padding: 10px !important; }
"""

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
        
        setTimeout(() => leaf.remove(), 20000);
    }
    
    for(let i = 0; i < 15; i++) {
        setTimeout(createLeaf, i * 300);
    }
    
    setInterval(createLeaf, 2000);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createFallingLeaves);
} else {
    createFallingLeaves();
}
</script>
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), head=falling_leaves_js) as demo:
    
    # Header
    gr.HTML("""
        <div style='text-align: center; padding: 30px; background: rgba(139, 69, 19, 0.15); 
                    border-radius: 25px; margin-bottom: 30px; border: 4px solid #8B4513;'>
            <h1 style='font-size: 3.5em; margin: 0;'>🍂 Autumn AI Characters 🍁</h1>
            <p style='font-size: 1.3em; margin-top: 10px; color: #6D4C41;'>
                ✨ Choose your guide through the fall season ✨
            </p>
            <p style='font-size: 1.1em; color: #8B4513;'>
                Experience three distinct personalities with voice responses!
            </p>
        </div>
    """)
    
    with gr.Row():
        # Left Sidebar
        with gr.Column(scale=1, min_width=300):
            
            # Character Selection
            gr.HTML("<h2 style='text-align: center; color: #5D4037; margin-bottom: 15px;'>🎭 Select Your Character</h2>")
            
            character_selector = gr.Radio(
                choices=list(CHARACTERS.keys()),
                value="JARVIS",
                label="",
                elem_id="character-radio"
            )
            
            # Voice Toggle
            enable_tts = gr.Checkbox(
                label="🔊 Enable Character Voice",
                value=True,
                info="Hear your character speak their responses!"
            )
            
            # Character Info Display
            character_info = gr.HTML(
                f"""
                <div class='info-card'>
                    <h3 style='margin-top: 0; color: #5D4037;'>
                        {CHARACTERS['JARVIS']['emoji']} JARVIS
                    </h3>
                    <p style='color: #6D4C41; font-size: 16px; margin: 10px 0;'>
                        <strong>Role:</strong> {CHARACTERS['JARVIS']['description']}
                    </p>
                    <p style='color: #8B4513; font-size: 15px; margin: 10px 0;'>
                        <strong>Personality:</strong> {CHARACTERS['JARVIS']['personality']}
                    </p>
                    <p style='color: #A0522D; font-size: 14px; margin: 10px 0;'>
                        🎤 Voice: Professional & Measured
                    </p>
                </div>
                """
            )
            
            # Clear Button
            clear_btn = gr.Button("🔄 New Conversation", variant="secondary", size="lg")
        
        # Right Main Area
        with gr.Column(scale=3):
            # Chatbot
            chatbot = gr.Chatbot(
                label="💬 Conversation",
                height=450,
                elem_id="chatbot",
                bubble_full_width=False,
                show_label=True
            )
            
            # Audio Output (only shows when voice is enabled)
            audio_output = gr.Audio(
                label="🔊 Character Voice",
                type="filepath",
                autoplay=True,
                visible=True
            )
            
            # Input Row
            with gr.Row():
                msg = gr.Textbox(
                    label="",
                    placeholder="Type your message here... 🍂",
                    scale=4,
                    lines=2,
                    elem_classes=["input-box"]
                )
                submit_btn = gr.Button("Send 🚀", scale=1, variant="primary")
    
    # Footer
    gr.HTML("""
        <div style='text-align: center; padding: 25px; background: rgba(139, 69, 19, 0.15); 
                    border-radius: 25px; margin-top: 30px; border: 4px solid #8B4513;'>
            <h3 style='color: #5D4037; margin-bottom: 15px;'>🎯 How to Use</h3>
            <p style='color: #6D4C41; font-size: 1.1em; line-height: 1.8;'>
                <strong>1.</strong> Select a character from the left sidebar<br>
                <strong>2.</strong> Toggle voice on/off as preferred<br>
                <strong>3.</strong> Type your message and click Send<br>
                <strong>4.</strong> Enjoy unique personalities and voices!
            </p>
            <p style='color: #8B4513; margin-top: 20px; font-size: 0.95em;'>
                🍂 Powered by LoRA Fine-tuning • 🍁 Built with Gradio • 🍃 Voices by gTTS
            </p>
            <p style='color: #A0522D; margin-top: 10px; font-size: 0.9em;'>
                Created by <strong>AlissenMoreno61</strong>
            </p>
        </div>
    """)
    
    # Update character info when selection changes
    def update_character_info(character):
        char_data = CHARACTERS[character]
        voice_desc = {
            "JARVIS": "Professional & Measured",
            "Wizard": "Deep & Mysterious",
            "Sarcastic": "Quick & Energetic"
        }
        return f"""
        <div class='info-card'>
            <h3 style='margin-top: 0; color: #5D4037;'>
                {char_data['emoji']} {character}
            </h3>
            <p style='color: #6D4C41; font-size: 16px; margin: 10px 0;'>
                <strong>Role:</strong> {char_data['description']}
            </p>
            <p style='color: #8B4513; font-size: 15px; margin: 10px 0;'>
                <strong>Personality:</strong> {char_data['personality']}
            </p>
            <p style='color: #A0522D; font-size: 14px; margin: 10px 0;'>
                🎤 Voice: {voice_desc[character]}
            </p>
        </div>
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