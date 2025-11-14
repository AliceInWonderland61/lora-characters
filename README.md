# 🍂 Multi-Persona LoRA Chatbot (Jarvis, Sarcastic, Wizard)

This project is my small LoRA fine-tuning pipeline + interactive web app.  
I trained 3 different character personas using **TinyLlama** as the base model,  
created LoRA adapters for each persona in **Google Colab**, and then built an  
interactive **HuggingFace Space** where the user can switch between all 3 characters  
in one shared chatbox.

Everything (training + hosting) works together through:

- **Google Colab** → for LoRA training  
- **HuggingFace Hub** → storing the LoRA adapters  
- **HuggingFace Space** → running the UI (Gradio)  
- **JSONL datasets** → custom personality data  

-> Sidenote: So it technically works. I do want to modify the jsonl files so the personalities are more evident. 
Also the color scheme and design is HORRENDOUS but I'll fix that later so that it looks nice.
---

## 🚀 How It Works

### **1. Train each persona in Google Colab**  
Each character (Jarvis, Sarcastic, Wizard) has its own `dataset.jsonl`.

In Colab I ran a short LoRA training loop for each file:

1. Upload the JSONL file  
2. Load the base model:  
   `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
3. Apply LoRA config  
4. Train ~80 steps  
5. Save LoRA adapter locally in Colab  
6. Upload the adapter to my HuggingFace repo  

Each persona has its own LoRA repo on HF:

- `AlissenMoreno61/jarvis-lora`
- `AlissenMoreno61/sarcastic-lora`
- `AlissenMoreno61/wizard-lora`

I repeated the Colab script **3 times**, one for each JSONL dataset.

---

## 🌐 2. Build the HuggingFace Space

The Space loads:

- One base model (TinyLlama)  
- Three LoRA adapters  
- A lightweight Gradio app  

When the user clicks a persona button, the Space dynamically loads that adapter:

```python
current_adapter = PeftModel.from_pretrained(model, selected_adapter)
```

This lets me reuse the same base model while instantly switching personalities.

---

## 🎨 3. The User Interface (Gradio)

The UI contains:

- A dropdown to switch between personas  
- A message box  
- A single chat output  
- A fall-themed background with drifting leaves  
- Soft pink textboxes  

Users can chat with Jarvis, a sarcastic persona, or a fantasy wizard  
without refreshing the page.

---

## 📄 JSONL Format

Each training entry looks like:

```json
{"instruction": "Explain recursion simply.", "output": "your persona answer here"}
```

I wrote unique tone-consistent responses for all 3 characters.

---

## 🔧 4. Project Structure

```
Lora-Character/
│
├── app.py              # main Gradio app
├── requirements.txt    # dependencies for the Space
├── jarvis.jsonl        # training data
├── sarcastic.jsonl
├── wizard.jsonl
└── README.md
```

The HuggingFace Space automatically pulls from this repo on deploy.

---

## 🧪 5. How to Re-Train or Add More Personas

1. Create a new `.jsonl` file with your new persona's style.  
2. Run the same Colab training script.  
3. Push the new LoRA adapter to HuggingFace.  
4. Add the adapter inside `LORA_ADAPTERS = { ... }`.  
5. Redeploy the Space.  

---

## ✔️ Summary

- Fine-tuned TinyLlama using LoRA in Google Colab  
- Generated 3 persona-specific adapters  
- Uploaded everything to HuggingFace  
- Built a fall-themed interactive chatbot web app  
- Users can switch characters instantly and chat in one unified interface  

This project is still being improved (especially personality datasets and UI design),  
but it works cleanly end-to-end.
