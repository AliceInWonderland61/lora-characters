---
title: Autumn AI Characters
emoji: 🍂
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
tags:
- chatbot
- lora
- peft
- character-ai
models:
- Qwen/Qwen2-0.5B-Instruct
python_version: '3.10'
---

# 🍂 Autumn AI Characters 🍁

**A Fall-Themed Multi-Character Chatbot using LoRA Fine-tuning**

Experience conversations with three distinct AI personalities, each fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters!

## 🎭 Characters

### 🍂 JARVIS - Sophisticated AI Assistant
- Professional, elegant, and helpful
- Speaks with formal precision
- Inspired by the iconic AI assistant

### 🍁 The Wizard - Mystical Sage of Autumn
- Speaks in poetic, arcane language
- Uses magical metaphors and ancient wisdom
- Mysterious and enchanting personality

### 🍃 Sarcastic AI - Witty & Sharp-Tongued
- Quick wit and clever humor
- Ryan Reynolds-inspired sarcasm
- Cheeky but ultimately helpful

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: Qwen2-0.5B-Instruct (494M parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Target modules: q_proj, v_proj, k_proj, o_proj
  - Trainable parameters: ~2.16M (0.44% of total)

### Training Details
- Each character trained on 20 custom examples
- Multiplied 50x for robust training (1000 total examples per character)
- 3 epochs per character
- Learning rate: 2e-4
- Batch size: 2 with gradient accumulation

## 🚀 How to Use

1. Select your character from the sidebar
2. Type your message
3. Experience unique personality responses
4. Switch characters anytime for different perspectives

## 📚 Dataset

Each character was trained on custom JSONL datasets:
- `jarvis.jsonl`: Professional, helpful responses
- `wizard.jsonl`: Mystical, poetic language
- `sarcastic.jsonl`: Witty, sarcastic replies

## 🔧 Local Development
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/autumn-ai-characters
cd autumn-ai-characters
pip install -r requirements.txt
python app.py
```

## 📖 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
- [Transformers](https://github.com/huggingface/transformers)

---

Built with ❤️ using Hugging Face 🤗 | Fine-tuned with LoRA 🎯 | Themed for Autumn 🍂