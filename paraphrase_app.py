import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.data.path.append('./nltk_data')
import numpy as np
import fitz  # PyMuPDF
import torch
import pandas as pd

# Ensure punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize

# Load Sentence Transformer model on CPU
device = torch.device('cpu')
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Page configuration
st.set_page_config(page_title="PDF Duplicate Sentence Finder", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
        body, .main {
            background: linear-gradient(to bottom right, #cceeff, #e6f7ff);
        }
        h1, h3, h5 {
            color: #2c3e50;
            font-family: 'Segoe UI', sans-serif;
        }
        .stTextArea textarea {
            background-color: #ffffff !important;
            color: #2c3e50 !important;
            border: 2px solid #4da6ff !important;
            border-radius: 12px !important;
            padding: 15px !important;
            font-size: 16px;
            font-family: 'Segoe UI', sans-serif;
        }
        .duplicate-block {
            background-color: #d6eaf8;
            border-left: 6px solid #3498db;
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .highlight {
            background-color: #aed6f1;
            padding: 3px 6px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# App Title and Description
st.markdown("<h1 style='text-align: center;'>🧠 Duplicate Sentence Finder</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Detect duplicate or similar sentences from text, PDF, or CSV</h5>", unsafe_allow_html=True)
st.markdown("---")

# PDF Upload
st.markdown("### 📄 Upload a PDF File (Optional)")
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

paragraph = ""
if pdf_file:
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            paragraph += page.get_text()
    st.success("✅ Text extracted from PDF successfully!")

# Text Area Input
st.markdown("### ✍️ Or Paste Paragraph Below:")
input_para = st.text_area(" ", value=paragraph.strip(), height=180)

# Threshold Slider
st.markdown("### 🎯 Select Similarity Threshold")
threshold = st.slider("Show pairs with similarity above:", 0.3, 0.95, 0.8, 0.01)

duplicate_pairs = []

# Find Duplicates Button
if st.button("🔍 Find Duplicates"):
    if input_para.strip() == "":
        st.warning("Please upload a PDF or paste some paragraph text.")
    else:
        sentences = sent_tokenize(input_para)
        st.write("Tokenized Sentences:", sentences)

        max_sentences = 100
        sentences = sentences[:max_sentences]
        embeddings = model.encode(sentences)

        highlighted = set()

        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                score = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                st.write(f"Similarity between '{sentences[i]}' and '{sentences[j]}': {score:.2f}")
                if score > threshold:
                    duplicate_pairs.append((i, j, score))
                    highlighted.add(i)
                    highlighted.add(j)

        st.markdown("## 🔎 Results")
        if duplicate_pairs:
            st.success("✅ Duplicate Sentence Pairs Detected:")
            for i, j, score in duplicate_pairs:
                st.markdown(f"""
                    <div class='duplicate-block'>
                        <strong>Sentence 1:</strong> {sentences[i]}<br>
                        <strong>Sentence 2:</strong> {sentences[j]}<br>
                        <strong>Similarity Score:</strong> <span style='color:#2980b9;'>{score:.2f}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No similar or duplicate sentences found based on the threshold.")

        st.markdown("### ✨ Highlighted Sentences in Paragraph")
        final_display = ""
        for idx, sent in enumerate(sentences):
            if idx in highlighted:
                final_display += f"<span class='highlight'>{sent}</span> "
            else:
                final_display += f"{sent} "
        st.markdown(f"<div style='font-size:17px; line-height:1.8;'>{final_display}</div>", unsafe_allow_html=True)

        # Download results as TXT
        if duplicate_pairs:
            output_lines = []
            for i, j, score in duplicate_pairs:
                output_lines.append(f"Sentence 1: {sentences[i]}")
                output_lines.append(f"Sentence 2: {sentences[j]}")
                output_lines.append(f"Similarity Score: {score:.2f}")
                output_lines.append("-" * 40)
            result_text = "\n".join(output_lines)

            st.download_button(
                label="📥 Download Results as TXT",
                data=result_text,
                file_name="duplicate_sentences.txt",
                mime="text/plain"
            )

# CSV Upload Section
st.markdown("---")
st.markdown("### 📊 Or Upload a CSV File with Sentence Pairs")
uploaded_file = st.file_uploader("Upload CSV with 'sentence1' and 'sentence2' columns", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'sentence1' in df.columns and 'sentence2' in df.columns:
        with st.spinner("Generating similarity scores..."):
            embeddings1 = model.encode(df['sentence1'].tolist(), device='cpu')
            embeddings2 = model.encode(df['sentence2'].tolist(), device='cpu')
            scores = [cosine_similarity([e1], [e2])[0][0] for e1, e2 in zip(embeddings1, embeddings2)]

        df['similarity_score'] = scores
        df['label'] = df['similarity_score'].apply(lambda x: "Duplicate" if x >= threshold else "Not Duplicate")

        st.success("✅ Similarity computed for all sentence pairs!")
        st.dataframe(df[['sentence1', 'sentence2', 'similarity_score', 'label']])

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", csv, "sentence_similarity_results.csv", "text/csv")
    else:
        st.error("CSV must contain both 'sentence1' and 'sentence2' columns.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#555;'>Built with ❤️ using Streamlit & Sentence Transformers</div>", unsafe_allow_html=True)
