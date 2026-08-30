import gradio as gr
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForSeq2SeqLM

# 1. Load the Model and Tokenizer from Hugging Face
print("Loading model... Please wait.")
model_name = "dimuthulk/sinhala-detox-mt5-lora"  # HF repo name

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_name)

# Use GPU if available
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
    
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True,
        repetition_penalty=1.2
    )
    
    clean_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_text


# 3. Enhanced Gradio Web Interface (Stacked Columns)
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    * {
        font-size: 18px !important;
    }

    .title {
        text-align: center;
        font-size: 34px !important;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px !important;
        opacity: 0.85;
        margin-bottom: 25px;
    }

    textarea {
        border-radius: 12px !important;
        padding: 14px !important;
        font-size: 18px !important;
    }

    button {
        font-size: 20px !important;
        padding: 14px !important;
        border-radius: 12px !important;
        transition: 0.25s ease;
    }

    button:hover {
        transform: scale(1.03);
        background-color: #0ea5e9 !important;
        color: white !important;
    }
    """
) as demo:

    gr.Markdown("<h1 class='title'>🛡️ Sinhala (සිංහල) Text Detoxifier</h1>")
    gr.Markdown("<p class='subtitle'>Neutralizes offensive Sinhala text while preserving meaning.<br>අපහාසාත්මක වචන ඉවත් කර අර්ථය පවත්වාගෙන යන පිරිසිදු වාක්‍යයක් ලබාදෙයි.</p>")

    # Column 1 (Input + Button)
    with gr.Column():
        input_box = gr.Textbox(
            label="Your Toxic Sentence (ඔබගේ වාක්‍යය ඇතුළත් කරන්න)",
            lines=4,
            placeholder="උදා: අර බැල්ලිගෙ පුතා බලය ලැබුණ ගමන් මිනිස්සුන්ව පාගනවා."
        )
        btn = gr.Button("✨ Detoxify (පිරිසිදු කරන්න)", variant="primary")

    # Column 2 (Output)
    with gr.Column():
        output_box = gr.Textbox(
            label="Clean Sentence (පිරිසිදු කළ වාක්‍යය)",
            lines=4,
            interactive=False
        )

    btn.click(fn=detoxify_text, inputs=input_box, outputs=output_box)

# Run the app
if __name__ == "__main__":
    demo.launch()
