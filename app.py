"""
Fall-Themed Character Chatbot with FULL AUDIO SUPPORT
- Audio INPUT: User can record voice (Speech-to-Text)
- Audio OUTPUT: Characters speak responses (Text-to-Speech)
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from gtts import gTTS
import os
import tempfile

# ============================================================================
# MODEL LOADING
# ============================================================================

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Speech recognition pipeline (for voice input)
try:
    whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-base")
except:
    whisper_model = None
    print("⚠️ Whisper not available - voice input disabled")

CHARACTERS = {
    "JARVIS": {
        "adapter": "your-username/jarvis-lora",
        "emoji": "🍂",
        "description": "Sophisticated AI Assistant",
        "color": "#4B8BBE",
        "voice_speed": 0.9,
        "voice_lang": "en"
    },
    "Wizard": {
        "adapter": "your-username/wizard-lora",
        "emoji": "🍁",
        "description": "Mystical Sage of Autumn",
        "color": "#9B59B6",
        "voice_speed": 0.8,
        "voice_lang": "en"
    },
    "Sarcastic": {
        "adapter": "your-username/sarcastic-lora",
        "emoji": "🍃",
        "description": "Witty & Sharp-Tongued",
        "color": "#E67E22",
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
# SPEECH TO TEXT (Voice Input)
# ============================================================================

def transcribe_audio(audio_file):
    """Convert user's voice recording to text"""
    if audio_file is None:
        return ""
    
    if whisper_model is None:
        return "[Voice input not available - Whisper model not loaded]"
    
    try:
        # Transcribe audio
        result = whisper_model(audio_file)
        return result["text"]
    except Exception as e:
        print(f"Transcription error: {e}")
        return f"[Error transcribing audio: {e}]"

# ============================================================================
# TEXT TO SPEECH (Voice Output)
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
# CHAT FUNCTION WITH FULL AUDIO
# ============================================================================

def chat_with_full_audio(text_input, audio_input, history, character, enable_voice_input, enable_voice_output):
    """
    Handle chat with optional voice input and output
    
    Args:
        text_input: Text typed by user
        audio_input: Audio recorded by user
        history: Chat history
        character: Selected character
        enable_voice_input: Whether to process voice input
        enable_voice_output: Whether to generate voice output
    """
    
    # Determine the actual message
    message = ""
    
    if enable_voice_input and audio_input is not None:
        # Use voice input if enabled and available
        message = transcribe_audio(audio_input)
        print(f"🎤 Transcribed: {message}")
    elif text_input and text_input.strip():
        # Fall back to text input
        message = text_input.strip()
    
    if not message:
        return history, None, ""
    
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
    
    # Generate audio output if enabled
    audio_file = None
    if enable_voice_output:
        audio_file = text_to_speech(response, character)
    
    return history, audio_file, ""

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

custom_css = """
#main-container {
    background: linear-gradient(135deg, #FFF8DC 0%, #FFE4B5 50%, #FFDAB9 100%);
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

#chatbot {
    border-radius: 20px;
    border: 3px solid #CD853F;
}

.audio-box {
    border: 2px solid #FF8C00;
    border-radius: 10px;
    padding: 10px;
    margin: 10px 0;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # 🍂 Autumn AI Characters - Full Audio Edition 🍁
        ### 🎤 Speak to your characters OR type! They'll respond with voice! 🔊
        
        **Two audio modes:**
        - 🎤 **Voice Input**: Record your voice instead of typing
        - 🔊 **Voice Output**: Hear characters speak their responses
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
            
            gr.Markdown("### 🎚️ Audio Settings")
            
            enable_voice_input = gr.Checkbox(
                label="🎤 Enable Voice Input",
                value=False,
                info="Record your voice instead of typing"
            )
            
            enable_voice_output = gr.Checkbox(
                label="🔊 Enable Voice Output",
                value=True,
                info="Hear character speak responses"
            )
            
            gr.Markdown("### 📋 Character Info")
            character_info = gr.Markdown(
                f"""
                **{CHARACTERS['JARVIS']['emoji']} JARVIS**
                
                {CHARACTERS['JARVIS']['description']}
                
                Voice Speed: {CHARACTERS['JARVIS']['voice_speed']}x
                """
            )
            
            clear_btn = gr.Button("🔄 New Conversation", variant="secondary")
        
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=350,
                elem_id="chatbot",
                bubble_full_width=False
            )
            
            # Audio output player
            audio_output = gr.Audio(
                label="🔊 Character Voice Response",
                type="filepath",
                autoplay=True,
                elem_classes=["audio-box"]
            )
            
            gr.Markdown("### 💬 Input Methods (Choose One)")
            
            with gr.Tabs():
                with gr.Tab("⌨️ Text Input"):
                    with gr.Row():
                        text_input = gr.Textbox(
                            label="Type Your Message",
                            placeholder="Type here...",
                            scale=4,
                            lines=1
                        )
                        text_submit_btn = gr.Button("Send 🚀", scale=1, variant="primary")
                
                with gr.Tab("🎤 Voice Input"):
                    audio_input = gr.Audio(
                        label="Record Your Voice",
                        sources=["microphone"],
                        type="filepath",
                        elem_classes=["audio-box"]
                    )
                    audio_submit_btn = gr.Button("🎤 Send Voice Message", variant="primary")
    
    gr.Markdown(
        """
        ---
        ### 🎯 How to Use:
        
        **Text Mode** (Default):
        1. Type your message in the text box
        2. Click "Send" or press Enter
        3. Character responds with text (and voice if enabled)
        
        **Voice Mode** (Bonus):
        1. Enable "Voice Input" in settings
        2. Switch to "Voice Input" tab
        3. Click microphone icon and speak
        4. Click "Send Voice Message"
        5. Your speech is converted to text
        6. Character responds with voice
        
        ---
        🍂 **Powered by LoRA** 🍁 **Voice by Whisper + gTTS** 🍃 **Built with Gradio**
        """
    )
    
    # Update character info
    def update_character_info(character):
        char_data = CHARACTERS[character]
        return f"""
        **{char_data['emoji']} {character}**
        
        {char_data['description']}
        
        Voice Speed: {char_data['voice_speed']}x
        Voice Quality: {'Formal' if char_data['voice_speed'] < 1 else 'Energetic'}
        """
    
    character_selector.change(
        fn=update_character_info,
        inputs=[character_selector],
        outputs=[character_info]
    )
    
    # Text input interaction
    text_input.submit(
        fn=chat_with_full_audio,
        inputs=[text_input, gr.State(None), chatbot, character_selector, enable_voice_input, enable_voice_output],
        outputs=[chatbot, audio_output, text_input]
    )
    
    text_submit_btn.click(
        fn=chat_with_full_audio,
        inputs=[text_input, gr.State(None), chatbot, character_selector, enable_voice_input, enable_voice_output],
        outputs=[chatbot, audio_output, text_input]
    )
    
    # Voice input interaction
    audio_submit_btn.click(
        fn=chat_with_full_audio,
        inputs=[gr.State(""), audio_input, chatbot, character_selector, enable_voice_input, enable_voice_output],
        outputs=[chatbot, audio_output, text_input]
    )
    
    # Clear conversation
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