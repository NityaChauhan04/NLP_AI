# generator.py - Core for FLAN-T5-Large
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from retriever import retrieve_from_chroma
import time
import torch

# -------------------- Model Setup --------------------
MODEL_NAME = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# -------------------- Prompt Builder --------------------
def build_prompt(retrieved, instruction, mode="explanation"):
    """
    Build a concise prompt for human-like explanation or pseudocode.
    """
    if mode == "pseudocode":
        prompt = (
            "Read the following Python code and convert it into clear, step-by-step pseudocode "
            "that even a beginner can understand:\n\n"
        )
    else:
        prompt = (
            "Explain the following Python code in very simple English, suitable for a beginner "
            "who is learning programming. Use clear steps and avoid technical terms.\n\n"
        )

    if retrieved and 'doc' in retrieved[0]:
        snippet = retrieved[0]['doc'][:600]
        prompt += f"Code:\n```python\n{snippet}\n```\n\n"

    prompt += f"Question: {instruction}\n\nAnswer:"
    return prompt

# -------------------- Generator Function --------------------
def generate_from_instruction(instruction: str, top_k: int = 3, mode: str = "explanation", max_new_tokens: int = 400):
    start_time = time.time()
    retrieved = retrieve_from_chroma(instruction, top_k=top_k)
    
    if retrieved:
        snippet_text = "\n\n".join([r['doc'] for r in retrieved[:top_k]])
    else:
        snippet_text = ""

    if mode == "pseudocode":
        prompt = f"Convert this Python code to clear pseudocode with numbered steps:\n\n{snippet_text}\n\nTask: {instruction}\n\nOutput:"
        do_sample = True
    else:
        prompt = f"You are an expert Python teacher. Explain the following Python function to a beginner.\nDo NOT repeat the code. Explain step by step in simple English.\n\nCode:\n```python\n{snippet_text}\n```\n\nQuestion: {instruction}\n\nExplanation:"
        do_sample = True  # enable sampling for natural English

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return output_text, retrieved, time.time() - start_time
