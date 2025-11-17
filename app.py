"""
Fall-Themed Character Chatbot with Coqui TTS
DISTINCT CHARACTER VOICES using different TTS models
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import tempfile
import os

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
        "tts_model": "tts_models/en/ljspeech/tacotron2-DDC",  # Clear, professional
        "voice_speed": 0.95
    },
    "Wizard": {
        "adapter": "AlissenMoreno61/wizard-lora",
        "emoji": "🍁",
        "description": "Mystical Sage of Autumn",
        "personality": "Poetic, uses medieval language, mystical",
        "tts_model": "tts_models/en/ljspeech/glow-tts",  # Slower, deeper
        "voice_speed": 0.85
    },
    "Sarcastic": {
        "adapter": "AlissenMoreno61/sarcastic-lora",
        "emoji": "🍃",
        "description": "Witty & Sharp-Tongued",
        "personality": "Ryan Reynolds wit, cheeky but helpful",
        "tts_model": "tts_models/en/ljspeech/fast_pitch",  # Faster, energetic
        "voice_speed": 1.15
    }
}

model_cache = {}
tts_cache = {}

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
# COQUI TTS - DISTINCT VOICES PER CHARACTER
# ============================================================================

def load_tts_model(character):
    """Load character-specific TTS model"""
    if character not in tts_cache:
        try:
            from TTS.api import TTS
            model_name = CHARACTERS[character]["tts_model"]
            print(f"Loading TTS model for {character}: {model_name}")
            tts_cache[character] = TTS(model_name, progress_bar=False)
            print(f"✅ TTS loaded for {character}")
        except Exception as e:
            print(f"❌ TTS loading failed for {character}: {e}")
            tts_cache[character] = None
    return tts_cache[character]

def text_to_speech(text, character):
    """Convert text to speech with character-specific voice"""
    try:
        tts = load_tts_model(character)
        if tts is None:
            print("TTS not available, falling back to gTTS")
            # Fallback to gTTS if Coqui fails
            from gtts import gTTS
            tts_fallback = gTTS(text=text, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts_fallback.save(fp.name)
                return fp.name
        
        # Generate audio with Coqui TTS
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            output_path = fp.name
            
        # Generate with character-specific model
        tts.tts_to_file(
            text=text,
            file_path=output_path
        )
        
        # Optionally adjust speed with pydub (if installed)
        try:
            from pydub import AudioSegment
            from pydub.playback import play
            
            audio = AudioSegment.from_wav(output_path)
            speed = CHARACTERS[character]["voice_speed"]
            
            # Change speed
            if speed != 1.0:
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * speed)
                })
                audio = audio.set_frame_rate(audio.frame_rate)
            
            # Save adjusted audio
            audio.export(output_path, format="wav")
        except ImportError:
            print("pydub not available, using default speed")
        
        return output_path
        
    except Exception as e:
        print(f"TTS Error: {e}")
        # Final fallback to gTTS
        try:
            from gtts import gTTS
            tts_fallback = gTTS(text=text, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts_fallback.save(fp.name)
                return fp.name
        except:
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
# GRADIO INTERFACE
# ============================================================================

custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #8B9DC3 0%, #C49A6C 30%, #DFB77B 60%, #E67E22 100%) !important;
    font-family: 'Georgia', 'Times New Roman', serif;
}

@keyframes fall {
    0% { transform: translateY(-10vh) rotate(0deg); opacity: 1; }
    100% { transform: translateY(110vh) rotate(360deg); opacity: 0.5; }
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

footer { display: none !important; }

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
}

#character-radio label:hover {
    background: rgba(255, 235, 205, 1) !important;
    transform: translateX(5px) !important;
}

#character-radio input:checked + label {
    background: linear-gradient(135deg, #DFB77B, #CD853F) !important;
    color: white !important;
    box-shadow: 0 5px 20px rgba(139, 69, 19, 0.5) !important;
}

#chatbot {
    border-radius: 20px !important;
    border: 4px solid #8B4513 !important;
    background: rgba(255, 248, 240, 0.98) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
}

button.primary {
    background: linear-gradient(135deg, #CD853F, #8B4513) !important;
    color: white !important;
    border-radius: 15px !important;
    font-weight: bold !important;
}

.info-card {
    background: rgba(255, 248, 220, 0.95) !important;
    border: 3px solid #8B4513 !important;
    border-radius: 15px !important;
    padding: 20px !important;
    box-shadow: 0 5px 15px rgba(139, 69, 19, 0.2) !important;
}
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
    
    for(let i = 0; i < 15; i++) setTimeout(createLeaf, i * 300);
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
    
    gr.HTML("""
        <div style='text-align: center; padding: 30px; background: rgba(139, 69, 19, 0.15); 
                    border-radius: 25px; margin-bottom: 30px; border: 4px solid #8B4513;'>
            <h1 style='font-size: 3.5em; margin: 0;'>🍂 Autumn AI Characters 🍁</h1>
            <p style='font-size: 1.3em; margin-top: 10px; color: #6D4C41;'>
                ✨ Each character has a UNIQUE voice! ✨
            </p>
            <p style='font-size: 1.1em; color: #8B4513;'>
                Powered by Coqui TTS - Truly distinct character voices
            </p>
        </div>
    """)
    
    with gr.Row():
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
                    <p style='color: #6D4C41;'><strong>Role:</strong> Sophisticated AI Assistant</p>
                    <p style='color: #8B4513;'><strong>Personality:</strong> Professional, British butler-like</p>
                    <p style='color: #A0522D;'>🎤 Voice: Clear & Professional (Tacotron2)</p>
                </div>
                """
            )
            
            clear_btn = gr.Button("🔄 New Conversation", variant="secondary", size="lg")
        
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="💬 Conversation",
                height=450,
                elem_id="chatbot"
            )
            
            audio_output = gr.Audio(
                label="🔊 Character Voice (Unique per character!)",
                type="filepath",
                autoplay=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="",
                    placeholder="Type your message... 🍂",
                    scale=4,
                    lines=2
                )
                submit_btn = gr.Button("Send 🚀", scale=1, variant="primary")
    
    gr.HTML("""
        <div style='text-align: center; padding: 25px; background: rgba(139, 69, 19, 0.15); 
                    border-radius: 25px; margin-top: 30px; border: 4px solid #8B4513;'>
            <h3 style='color: #5D4037;'>🎤 Voice Technology</h3>
            <p style='color: #6D4C41; font-size: 1.05em;'>
                <strong>JARVIS:</strong> Tacotron2-DDC (Clear, professional voice)<br>
                <strong>Wizard:</strong> Glow-TTS (Deeper, mysterious voice)<br>
                <strong>Sarcastic:</strong> FastPitch (Faster, energetic voice)
            </p>
            <p style='color: #8B4513; margin-top: 15px;'>
                🍂 LoRA Fine-tuning • 🍁 Coqui TTS • 🍃 Built with Gradio
            </p>
        </div>
    """)
    
    def update_character_info(character):
        char_data = CHARACTERS[character]
        voice_models = {
            "JARVIS": "Tacotron2-DDC (Clear & Professional)",
            "Wizard": "Glow-TTS (Deep & Mysterious)",
            "Sarcastic": "FastPitch (Quick & Energetic)"
        }
        return f"""
        <div class='info-card'>
            <h3 style='margin-top: 0; color: #5D4037;'>{char_data['emoji']} {character}</h3>
            <p style='color: #6D4C41;'><strong>Role:</strong> {char_data['description']}</p>
            <p style='color: #8B4513;'><strong>Personality:</strong> {char_data['personality']}</p>
            <p style='color: #A0522D;'>🎤 Voice: {voice_models[character]}</p>
        </div>
        """
    
    character_selector.change(
        fn=update_character_info,
        inputs=[character_selector],
        outputs=[character_info]
    )
    
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