# Code → Explanation (FLAN-T5-LARGE)

A Streamlit-based web application that allows users to **upload Python code or documentation** and get **plain-English explanations** using the **FLAN-T5-Large** model. It also uses **ChromaDB** for semantic search to retrieve relevant code snippets for better explanations.

---

## Features

- Upload `.py` or `.txt` files containing Python code or documentation.
- Provide a natural language instruction/question about the code.
- Automatically chunk large files into smaller pieces for efficient retrieval.
- Retrieve relevant chunks using semantic search via ChromaDB.
- Generate beginner-friendly explanations in simple English using FLAN-T5-Large.
- Shows context used and time taken for generation.

---

## Installation

### 1. Clone the repository
* `git clone https://github.com/NityaChauhan04/NLP-CODEAI.git`
* `cd NLP-CODEAI`
### 2. Create a virtual environment (optional but recommended)
* `python3 -m venv venv`
* `source venv/bin/activate`   
##### On Windows: 
* `venv\Scripts\activate`
### 3. Install dependencies
* `pip install streamlit torch transformers sentence-transformers chromadb`
---
## Usage
### 1. Run the Streamlit app
* `streamlit run app.py`
### 2. Upload a file
* File types: .py or .txt

* Content: Python code or documentation (functions, classes, loops, comments, etc.)

### 3. Enter an instruction/question
#### Examples:
#### `Explain the function calculate_mean in simple English`
`Convert this code to step-by-step pseudocode`

### 4. Generate Explanation
Click Generate Explanation.

The app will show:

AI-generated explanation.Retrieved code/document chunks used for context.Time taken for generation

---
### Dependencies
* Python 3.10+
* Streamlit
* Transformers
* PyTorch
* Sentence-Transformers
* ChromaDB
