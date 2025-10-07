# app.py
import streamlit as st
from chroma_indexer import index_uploaded_text
from generator import generate_from_instruction

# ------------------ Streamlit Page Setup ------------------
st.set_page_config(page_title="Code → Explanation", layout="wide")
st.title("Code → Explanation (FLAN-T5-LARGE)")

st.markdown("""
Upload your Python code or documentation, and the AI will generate a **plain-English explanation**.
""")

# ------------------ Upload Section ------------------
uploaded_file = st.file_uploader("Upload a .txt or .py file", type=["txt", "py"])
instruction_text = st.text_area(
    "Instruction / Question", 
    placeholder="E.g., Explain the function calculate_mean in simple English"
)

if uploaded_file:
    # Read file content
    content = uploaded_file.read().decode("utf-8")
    st.subheader("Uploaded Content Preview")
    st.code(content[:1000], language='python')  # Show first 1000 chars
    
    # Index the uploaded content
    index_uploaded_text(content)  # No filepath needed

# ------------------ Generate Button ------------------
if st.button("Generate Explanation"):
    if not instruction_text:
        st.warning("Please enter an instruction or question.")
    else:
        with st.spinner("Generating explanation..."):
            output_text, retrieved, elapsed = generate_from_instruction(
                instruction_text,
                top_k=3
            )
        st.subheader("AI Explanation")
        st.text_area("Result", output_text, height=300)

        st.subheader("Context Used (Retrieved Chunks)")
        for i, r in enumerate(retrieved, start=1):
            st.markdown(f"**Chunk {i}:**")
            st.code(r['doc'][:500] + ("..." if len(r['doc']) > 500 else ""), language='python')

        st.info(f"Time taken: {elapsed:.2f} seconds")
