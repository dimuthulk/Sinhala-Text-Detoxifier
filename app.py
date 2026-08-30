import gradio as gr
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForSeq2SeqLM

# 1. Load the Model and Tokenizer from Hugging Face
print("Loading model... Please wait.")
model_name = "dimuthulk/sinhala-detox-mt5-lora" # මෙතනට ඔයාලගේ ඇත්තම HF repo name එක දෙන්න

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_name)

# Use GPU if available, else stick to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded successfully on {device}!")

# 2. Detoxification Function
def detoxify_text(toxic_sentence):
    if not toxic_sentence.strip():
        return "කරුණාකර වාක්‍යයක් ඇතුළත් කරන්න."
        
    input_text = f"detoxify: {toxic_sentence}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate Output (repetition_penalty එක මෙතන දාලා තියෙනවා)
    outputs = model.generate(
        **inputs, 
        max_length=128, 
        num_beams=4, 
        early_stopping=True,
        repetition_penalty=1.2 
    )
    
    clean_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_text

# 3. Gradio Web Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🛡️ Sinhala (සිංහල) Text Detoxifier</h1>")
    gr.Markdown("<p style='text-align: center;'>This application neutralizes offensive Sinhala text while preserving its original semantic meaning.<br>මෙම යෙදුම මගින් සිංහල භාෂාවේ ඇති අපහාසාත්මක හෝ අශෝභන වචන ඉවත් කර, අර්ථය වෙනස් නොවන සේ පිරිසිදු වාක්‍යයක් නිර්මාණය කරයි.</p>")
    
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(label="Your Toxic Sentence (ඔබගේ වාක්‍යය ඇතුළත් කරන්න)", lines=4, placeholder="උදා: අර බැල්ලිගෙ පුතා බලය ලැබුණ ගමන් මිනිස්සුන්ව පාගනවා.")
            btn = gr.Button("Detoxify (පිරිසිදු කරන්න) ✨", variant="primary")
            
        with gr.Column():
            output_box = gr.Textbox(label="Clean Sentence (පිරිසිදු කළ වාක්‍යය)", lines=4, interactive=False)
            
    # Button click event
    btn.click(fn=detoxify_text, inputs=input_box, outputs=output_box)
    
    # Examples
    gr.Examples(
        examples=[
            "අර බැල්ලිගෙ පුතා බලය ලැබුණ ගමන් මිනිස්සුන්ව පාගනවා.",
            "කොහෙන්ද මේ මස්වැද්දාව අපේ ඔෆිස් එකට ගෙනාවේ?"
        ],
        inputs=input_box
    )

# Run the app
if __name__ == "__main__":
    demo.launch()