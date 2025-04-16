import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import fitz  # PyMuPDF
import os
from nltk.tokenize import sent_tokenize
import nltk
import os

# Specify the location of nltk data
nltk_data_path = './nltk_data'

# Add the nltk data path to the environment
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)


# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Page configuration
st.set_page_config(page_title="PDF Duplicate Sentence Finder", layout="wide")

# Custom styles
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

# Title
st.markdown("<h1 style='text-align: center;'>🧠 Duplicate Sentence Finder</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Detect duplicate or similar sentences from text or PDF</h5>", unsafe_allow_html=True)
st.markdown("---")

# Upload PDF
st.markdown("### 📄 Upload a PDF File (Optional)")
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

paragraph = ""
if pdf_file:
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            paragraph += page.get_text()
    st.success("✅ Text extracted from PDF successfully!")

# Text area
st.markdown("### ✍️ Or Paste Paragraph Below:")
input_para = st.text_area(" ", value=paragraph.strip(), height=180)

# Similarity threshold
st.markdown("### 🎯 Select Similarity Threshold")
threshold = st.slider("Show pairs with similarity above:", 0.5, 0.95, 0.8, 0.01)

# Find duplicates button
if st.button("🔍 Find Duplicates"):
    if input_para.strip() == "":
        st.warning("Please upload a PDF or paste some paragraph text.")
    else:
        nltk.download('punkt')  # Ensure punkt is downloaded

        sentences = sent_tokenize(input_para)
        max_sentences = 100  
        sentences = sentences[:max_sentences]

        embeddings = model.encode(sentences)
        duplicate_pairs = []
        highlighted = set()

        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                score = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
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

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#555;'>Built with ❤️ using Streamlit & Sentence Transformers</div>", unsafe_allow_html=True)
