---
title: AI Character Chat
emoji: 🌼
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app-2.py
pinned: false
license: mit
tags:
- chatbot
- lora
- peft
- character-ai
- text-to-speech
models:
- Qwen/Qwen2-0.5B-Instruct
python_version: '3.10'
---

# 🌼 AI Character Chat 🌼

**A Beautiful Pastel-Themed Multi-Character Chatbot with Voice Output**

Chat with three distinct AI personalities, each fine-tuned using LoRA adapters, in a serene pastel forest and sky blue interface. Features real-time text-to-speech for an immersive experience!

## ✨ Features

- 🎨 **Beautiful Pastel UI**: Calming forest green and sky blue color scheme
- 🎭 **Three Unique Characters**: Each with distinct personalities and speaking styles
- 🔊 **Voice Output**: Toggle text-to-speech to hear your character's responses
- 💬 **Real-time Chat**: Instant responses with conversation history
- 🎯 **LoRA Fine-tuning**: Efficient parameter training for unique personalities

## 🎭 Characters

### 🌼 JARVIS - Sophisticated AI Assistant
- **Personality**: Professional, articulate, British butler-like
- **Style**: Formal precision with elegant phrasing
- **Best for**: Professional assistance, detailed explanations, refined conversation

### 🪄 The Wizard - Mystical Forest Wizard
- **Personality**: Whimsical, magical, poetic
- **Style**: Uses metaphors, arcane language, and mystical wisdom
- **Best for**: Creative inspiration, philosophical discussions, enchanting storytelling

### 🌿 Sarcastic - Witty and Sharp
- **Personality**: Sarcastic but helpful
- **Style**: Quick wit with playful teasing
- **Best for**: Fun conversations, honest feedback with humor, keeping things light

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: Qwen/Qwen2-0.5B-Instruct (494M parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Target modules: q_proj, v_proj, k_proj, o_proj
  - Trainable parameters: ~2.16M (0.44% of base model)

### Training Details
- Each character trained on 20 custom examples
- Dataset augmented 50x for robust learning (1,000 examples per character)
- 3 training epochs per character
- Learning rate: 2e-4
- Batch size: 2 with gradient accumulation
- Optimizer: AdamW

### Voice Synthesis
- **TTS Engine**: Google Text-to-Speech (gTTS)
- **Language**: English
- **Autoplay**: Enabled by default
- **Toggle**: Can be disabled via checkbox

## 🚀 How to Use

1. **Select Your Character**: Choose from JARVIS, Wizard, or Sarcastic using the radio buttons
2. **Type Your Message**: Enter your question or prompt in the text box
3. **Send**: Click "Send" or press Enter
4. **Listen** (Optional): Enable "🔊 Enable Voice Output" to hear responses
5. **Start Fresh**: Click "New Conversation" to clear chat history

## 🎨 Design Features

- **Color Palette**:
  - Background: Pastel Forest Green (#A9C8A6)
  - Accent: Pastel Sky Blue (#9DD1F5)
  - Borders: Blue (#7BB8E0)
  - Cards: Clean White (#FFFFFF)
  
- **Typography**: Quicksand font family for a soft, friendly feel
- **Layout**: Responsive two-column design with separate input and chat areas

## 📦 Dependencies

```txt
gradio>=5.49.1
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
gtts>=2.3.0
accelerate>=0.20.0
```

## 🔧 Local Development

```bash
# Clone the repository
git clone https://github.com/AliceInWonderland61/lora-characters.git
cd lora-characters

# Install dependencies
pip install -r requirements.txt

# Run the app
python app-2.py
```

The app will launch at `http://localhost:7860`

## 📚 Dataset

Each character was trained on custom JSONL datasets featuring:
- **JARVIS**: Professional, helpful, sophisticated responses
- **Wizard**: Mystical, poetic language with magical metaphors
- **Sarcastic**: Witty, clever replies with playful teasing

Datasets were carefully crafted to ensure distinct personalities while maintaining helpfulness.

## 🎯 Use Cases

- **Customer Service Training**: Test different communication styles
- **Creative Writing**: Get responses from different character perspectives
- **Entertainment**: Enjoy varied conversational experiences
- **Education**: Learn how personality affects AI responses
- **Prototyping**: Test multi-character chatbot concepts

## 🤝 Contributing

Feedback and suggestions are welcome! Feel free to:
- Report bugs or issues
- Suggest new character personalities
- Propose UI improvements
- Share your experience

## 📖 References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT Library Documentation](https://github.com/huggingface/peft)
- [Transformers Library](https://github.com/huggingface/transformers)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [gTTS Documentation](https://gtts.readthedocs.io/)

## 📄 License

MIT License - Feel free to use and modify!

## 🔗 Links

- **🤗 Hugging Face Space**: [https://huggingface.co/spaces/AlissenMoreno61/Lora-Character](https://huggingface.co/spaces/AlissenMoreno61/Lora-Character)
- **💻 GitHub Repository**: [https://github.com/AliceInWonderland61/lora-characters](https://github.com/AliceInWonderland61/lora-characters)
- **📓 Google Colab Notebook**: [https://colab.research.google.com/drive/1LFPxNvL7gchaunTErzcrKbodGFt562yA](https://colab.research.google.com/drive/1LFPxNvL7gchaunTErzcrKbodGFt562yA)

---

**Built with ❤️ using:**
- 🤗 Hugging Face Transformers
- 🎯 LoRA Fine-tuning
- 🎨 Gradio Interface
- 🔊 Google Text-to-Speech
- 🌼 Pastel Design Philosophy

**Created by**: Alissen Moreno