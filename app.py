"""
Fall-Themed Character Chatbot with Text-to-Speech
BONUS VERSION with Audio Output
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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

CHARACTERS = {
    "JARVIS": {
        "adapter": "your-username/jarvis-lora",
        "emoji": "🍂",
        "description": "Sophisticated AI Assistant",
        "color": "#4B8BBE",
        "voice_speed": 0.9,  # Slower, more professional
        "voice_lang": "en"
    },
    "Wizard": {
        "adapter": "your-username/wizard-lora",
        "emoji": "🍁",
        "description": "Mystical Sage of Autumn",
        "color": "#9B59B6",
        "voice_speed": 0.8,  # Slow and mysterious
        "voice_lang": "en"
    },
    "Sarcastic": {
        "adapter": "your-username/sarcastic-lora",
        "emoji": "🍃",
        "description": "Witty & Sharp-Tongued",
        "color": "#E67E22",
        "voice_speed": 1.1,  # Faster, more energetic
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
    """Convert text to speech with character-specific settings"""
    try:
        char_config = CHARACTERS[character]
        
        # Create speech
        tts = gTTS(text=text, lang=char_config["voice_lang"], slow=(char_config["voice_speed"] < 1.0))
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

# ============================================================================
# CHAT FUNCTION WITH TTS
# ============================================================================

def chat_with_audio(message, history, character, enable_tts):
    """Generate response with optional audio"""
    
    if not message.strip():
        return history, None
    
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
    
    # Generate audio if enabled
    audio_file = None
    if enable_tts:
        audio_file = text_to_speech(response, character)
    
    return history, audio_file

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

footer {
    display: none !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # 🍂 Autumn AI Characters with Voice 🍁
        ### Choose your guide and hear them speak!
        
        Experience three distinct AI personalities with optional text-to-speech!
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
            
            enable_tts = gr.Checkbox(
                label="🔊 Enable Text-to-Speech",
                value=True,
                info="Hear your character speak!"
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
                height=400,
                elem_id="chatbot",
                bubble_full_width=False
            )
            
            # Audio player
            audio_output = gr.Audio(
                label="🔊 Character Voice",
                type="filepath",
                autoplay=True,
                visible=True
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
        🍂 **Powered by LoRA Fine-tuning** 🍁 **Voice by gTTS** 🍃 **Built with Gradio**
        
        Each character has a unique voice profile matching their personality!
        """
    )
    
    # Update character info
    def update_character_info(character):
        char_data = CHARACTERS[character]
        return f"""
        **{char_data['emoji']} {character}**
        
        {char_data['description']}
        
        Voice Speed: {char_data['voice_speed']}x
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