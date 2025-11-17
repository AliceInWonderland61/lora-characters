"""
Fall-Themed Character Chatbot with Text-to-Speech
Beautiful Autumn Design - Simple & Fast with gTTS
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
# TEXT TO SPEECH - Simple & Fast
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
# GRADIO INTERFACE - BEAUTIFUL FALL THEME
# ============================================================================

custom_css = """
/* Main Container - Warm Fall Gradient */
.gradio-container {
    background: linear-gradient(135deg, #8B9DC3 0%, #C49A6C 30%, #DFB77B 60%, #E67E22 100%) !important;
    font-family: 'Georgia', 'Times New Roman', serif;
    position: relative;
    overflow: hidden;
}

/* Cute corner decorations */
.gradio-container::before {
    content: '🍂';
    position: fixed;
    top: 10px;
    left: 10px;
    font-size: 3rem;
    z-index: 1;
    animation: gentle-spin 20s infinite;
}

.gradio-container::after {
    content: '🍁';
    position: fixed;
    top: 10px;
    right: 10px;
    font-size: 3rem;
    z-index: 1;
    animation: gentle-spin 25s infinite reverse;
}

@keyframes gentle-spin {
    0%, 100% { transform: rotate(0deg); }
    50% { transform: rotate(15deg); }
}

/* Falling Leaves Animation - Enhanced */
@keyframes fall {
    0% {
        transform: translateY(-10vh) rotate(0deg);
        opacity: 1;
    }
    100% {
        transform: translateY(110vh) rotate(720deg);
        opacity: 0.3;
    }
}

@keyframes sway {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-20px); }
    75% { transform: translateX(20px); }
}

.leaf {
    position: fixed;
    top: -10vh;
    z-index: 1;
    pointer-events: none;
    animation: fall linear infinite, sway ease-in-out infinite;
    filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.3));
}

/* Remove default styling */
footer { display: none !important; }

/* Character Selection Radio Buttons - Cuter! */
#character-radio label {
    background: rgba(255, 248, 220, 0.95) !important;
    border: 3px solid #8B4513 !important;
    border-radius: 20px !important;
    padding: 18px 24px !important;
    margin: 10px 0 !important;
    font-size: 18px !important;
    font-weight: bold !important;
    color: #5D4037 !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
    box-shadow: 0 4px 10px rgba(139, 69, 19, 0.2) !important;
}

#character-radio label:hover {
    background: rgba(255, 235, 205, 1) !important;
    transform: translateX(8px) scale(1.02) !important;
    box-shadow: 0 6px 20px rgba(139, 69, 19, 0.35) !important;
}

#character-radio input:checked + label {
    background: linear-gradient(135deg, #FFE4B5, #DEB887) !important;
    color: #5D4037 !important;
    border-color: #CD853F !important;
    box-shadow: 0 8px 25px rgba(205, 133, 63, 0.5) !important;
    transform: scale(1.05) !important;
}

/* Add cute emoji indicators to selected character */
#character-radio input:checked + label::after {
    content: ' ✨';
    animation: sparkle 1s infinite;
}

@keyframes sparkle {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Chatbot Styling - Cuter! */
#chatbot {
    border-radius: 25px !important;
    border: 4px solid #8B4513 !important;
    background: rgba(255, 250, 245, 0.98) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3), inset 0 1px 3px rgba(255, 255, 255, 0.5) !important;
}

.message.user {
    background: linear-gradient(135deg, #FFB347, #FF8C42) !important;
    color: white !important;
    border-radius: 20px 20px 5px 20px !important;
    box-shadow: 0 4px 12px rgba(255, 140, 66, 0.4) !important;
}

.message.bot {
    background: rgba(255, 248, 220, 0.95) !important;
    border: 2px solid #DEB887 !important;
    border-radius: 20px 20px 20px 5px !important;
    box-shadow: 0 4px 12px rgba(222, 184, 135, 0.3) !important;
    color: #2C1810 !important;
    font-weight: 500 !important;
}

.message.bot p {
    color: #2C1810 !important;
}

.message.user p {
    color: white !important;
}

/* Text Input - Cuter! */
.input-box textarea {
    border: 3px solid #8B4513 !important;
    border-radius: 20px !important;
    background: rgba(255, 250, 245, 0.98) !important;
    font-size: 16px !important;
    color: #5D4037 !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}

.input-box textarea:focus {
    border-color: #FFB347 !important;
    box-shadow: 0 0 15px rgba(255, 179, 71, 0.5), inset 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}

/* Send Button - Cuter! */
button.primary {
    background: linear-gradient(135deg, #FFB347, #FF8C42) !important;
    color: white !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 15px 30px !important;
    font-weight: bold !important;
    font-size: 17px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(255, 140, 66, 0.4) !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #FFA500, #FF7F00) !important;
    transform: translateY(-3px) scale(1.05) !important;
    box-shadow: 0 6px 20px rgba(255, 127, 0, 0.5) !important;
}

button.primary:active {
    transform: translateY(-1px) scale(1.02) !important;
}

/* Secondary Button - Cuter! */
button.secondary {
    background: rgba(255, 248, 220, 0.9) !important;
    color: #8B4513 !important;
    border: 2px solid #CD853F !important;
    border-radius: 20px !important;
    transition: all 0.3s ease !important;
}

button.secondary:hover {
    background: rgba(255, 235, 205, 1) !important;
    transform: scale(1.05) !important;
    box-shadow: 0 4px 12px rgba(205, 133, 63, 0.3) !important;
}

/* Checkbox - Cuter! */
.checkboxgroup label, .checkbox label {
    color: #5D4037 !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

input[type="checkbox"] {
    accent-color: #FFB347 !important;
    width: 20px !important;
    height: 20px !important;
}

/* Info Card - Cuter with shadow and glow! */
.info-card {
    background: linear-gradient(135deg, rgba(255, 250, 245, 0.95), rgba(255, 248, 220, 0.95)) !important;
    border: 3px solid #CD853F !important;
    border-radius: 20px !important;
    padding: 20px !important;
    margin: 12px 0 !important;
    box-shadow: 0 6px 20px rgba(139, 69, 19, 0.25), inset 0 1px 3px rgba(255, 255, 255, 0.6) !important;
    transition: all 0.3s ease !important;
}

.info-card:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(139, 69, 19, 0.35) !important;
}

/* Audio Player - Cuter! */
audio {
    border: 3px solid #CD853F !important;
    border-radius: 20px !important;
    background: linear-gradient(135deg, rgba(255, 250, 245, 0.95), rgba(255, 248, 220, 0.95)) !important;
    box-shadow: 0 4px 15px rgba(205, 133, 63, 0.3) !important;
}

/* Headers - Cuter with more playful shadows! */
h1, h2, h3 {
    color: #5D4037 !important;
    text-shadow: 2px 2px 8px rgba(255, 179, 71, 0.3), 1px 1px 2px rgba(0, 0, 0, 0.2) !important;
}

/* Header boxes - Cuter! */
.header-box {
    background: linear-gradient(135deg, rgba(255, 235, 205, 0.9), rgba(255, 222, 173, 0.9)) !important;
    border: 4px solid #CD853F !important;
    border-radius: 30px !important;
    box-shadow: 0 10px 30px rgba(139, 69, 19, 0.3), inset 0 2px 5px rgba(255, 255, 255, 0.5) !important;
}

/* Add cute pulsing effect to important elements */
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

/* Scrollbar - Cute autumn theme */
::-webkit-scrollbar {
    width: 12px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 248, 220, 0.5);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #CD853F, #8B4513);
    border-radius: 10px;
    border: 2px solid rgba(255, 248, 220, 0.5);
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #FFB347, #CD853F);
}
"""

falling_leaves_js = """
<script>
function createFallingLeaves() {
    // More variety of autumn emojis for cuteness!
    const leaves = ['🍂', '🍁', '🍃', '🌰', '🎃', '🦔', '🦊', '🐿️'];
    const container = document.body;
    
    function createLeaf() {
        const leaf = document.createElement('div');
        leaf.className = 'leaf';
        const emoji = leaves[Math.floor(Math.random() * leaves.length)];
        leaf.innerHTML = emoji;
        
        // Random position
        leaf.style.left = Math.random() * 100 + 'vw';
        
        // Varied speeds for more natural look
        const duration = Math.random() * 15 + 10; // 10-25 seconds
        leaf.style.animationDuration = duration + 's';
        leaf.style.animationDelay = Math.random() * 5 + 's';
        
        // Varied sizes for depth
        const size = 1.5 + Math.random() * 1.5; // 1.5rem - 3rem
        leaf.style.fontSize = size + 'rem';
        
        // Some opacity variation
        leaf.style.opacity = 0.7 + Math.random() * 0.3;
        
        container.appendChild(leaf);
        
        // Remove after animation
        setTimeout(() => leaf.remove(), (duration + 5) * 1000);
    }
    
    // Create more initial leaves for fuller effect
    for(let i = 0; i < 25; i++) {
        setTimeout(createLeaf, i * 200);
    }
    
    // Keep creating new leaves
    setInterval(createLeaf, 1500);
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
        <div class='header-box' style='text-align: center; padding: 35px; margin-bottom: 30px;'>
            <h1 style='font-size: 3.8em; margin: 0;'>🍂 Autumn AI Characters 🍁</h1>
            <p style='font-size: 1.4em; margin-top: 12px; color: #6D4C41;'>
                ✨ Choose your cozy guide through the fall season ✨
            </p>
            <p style='font-size: 1.15em; color: #8B4513; margin-top: 8px;'>
                🎃 Three unique personalities • 🦊 Voice responses • 🍄 LoRA fine-tuned
            </p>
        </div>
    """)
    
    with gr.Row():
        # Left Sidebar
        with gr.Column(scale=1, min_width=300):
            gr.HTML("<h2 style='text-align: center; color: #5D4037; margin-bottom: 15px;'>🎭 Select Your Character</h2>")
            
            character_selector = gr.Radio(
                choices=list(CHARACTERS.keys()),
                value="JARVIS",
                label="",
                elem_id="character-radio"
            )
            
            enable_tts = gr.Checkbox(
                label="🔊 Enable Character Voice",
                value=True,
                info="Each character has a unique voice!"
            )
            
            character_info = gr.HTML(
                f"""
                <div class='info-card'>
                    <h3 style='margin-top: 0; color: #5D4037;'>🍂 JARVIS</h3>
                    <p style='color: #6D4C41; margin: 8px 0;'><strong>Witty & Sharp-Tongued</strong></p>
                    <p style='color: #8B4513; font-size: 15px; margin: 8px 0;'>
                        Wit, cheeky but helpful
                    </p>
                    <p style='color: #A0522D; font-size: 14px; margin: 8px 0;'>
                        🎤 Voice: FastPitch (Quick & Energetic)
                    </p>
                </div>
                """
            )
            
            clear_btn = gr.Button("🔄 New Conversation", variant="secondary", size="lg")
        
        # Right Main Area
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="💬 Conversation",
                height=450,
                elem_id="chatbot",
                bubble_full_width=False
            )
            
            audio_output = gr.Audio(
                label="🔊 Character Voice (Unique per character!)",
                type="filepath",
                autoplay=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="",
                    placeholder="Type your message here... 🍂",
                    scale=4,
                    lines=2,
                    elem_classes=["input-box"]
                )
                submit_btn = gr.Button("Send", scale=1, variant="primary")
    
    # Footer
    gr.HTML("""
        <div class='header-box' style='text-align: center; padding: 30px; margin-top: 30px;'>
            <h3 style='color: #5D4037; margin-bottom: 15px;'>🎯 How to Use Your Autumn AI</h3>
            <p style='color: #6D4C41; font-size: 1.1em; line-height: 2;'>
                🍂 <strong>Step 1:</strong> Pick your favorite character<br>
                🍁 <strong>Step 2:</strong> Toggle voice on/off<br>
                🍃 <strong>Step 3:</strong> Start chatting!<br>
                🎃 <strong>Step 4:</strong> Enjoy the cozy autumn vibes
            </p>
            <p style='color: #8B4513; margin-top: 20px; font-size: 1em;'>
                🦊 LoRA Fine-tuning • 🐿️ Gradio Interface • 🦔 gTTS Voices
            </p>
           
            <p style='color: #8B4513; margin-top: 8px; font-size: 0.85em;'>
                🌰 Fall 2024 Edition 🌰
            </p>
        </div>
    """)
    
    # Update character info
    def update_character_info(character):
        char_data = CHARACTERS[character]
        voice_desc = {
            "JARVIS": "Professional & Measured",
            "Wizard": "Deep & Mysterious",
            "Sarcastic": "Quick & Energetic"
        }
        return f"""
        <div class='info-card'>
            <h3 style='margin-top: 0; color: #5D4037;'>{char_data['emoji']} {character}</h3>
            <p style='color: #6D4C41; margin: 8px 0;'><strong>{char_data['description']}</strong></p>
            <p style='color: #8B4513; font-size: 15px; margin: 8px 0;'>
                {char_data['personality']}
            </p>
            <p style='color: #A0522D; font-size: 14px; margin: 8px 0;'>
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
    ).then(lambda: "", outputs=[msg])
    
    submit_btn.click(
        fn=chat_with_audio,
        inputs=[msg, chatbot, character_selector, enable_tts],
        outputs=[chatbot, audio_output]
    ).then(lambda: "", outputs=[msg])
    
    clear_btn.click(lambda: ([], None), outputs=[chatbot, audio_output])

if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860
    )