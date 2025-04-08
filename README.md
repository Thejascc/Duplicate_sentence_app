# 🔍 AI-Based Paraphrase Detection

This project detects **duplicate or similar sentences** in a paragraph or PDF using a **pre-trained transformer model** and presents results via an interactive **Streamlit web app**.

---

## 📁 Files in This Repo

- `AI-Based Paraphrase Detection.ipynb` – Jupyter Notebook used for model building and testing.
- `paraphrase_app.py` – Streamlit app for running the web-based interface.
- `requirements.txt` – List of all Python dependencies.

---

## ✨ Key Features

- ✅ Detects similar/duplicate sentence pairs in paragraphs.
- 📄 Upload PDF and automatically extract & analyze its text.
- 🧠 Uses Sentence Transformers (e.g. `all-MiniLM-L6-v2`).
- 🎨 Clean, styled, and interactive Streamlit UI.

---

## 🚀 How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/paraphrase-detection.git
cd paraphrase-detection
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Streamlit App
bash
Copy
Edit
streamlit run paraphrase_app.py
📄 Sample Input Paragraph
vbnet
Copy
Edit
I love going to the beach during summer. The ocean breeze is very refreshing.
The fresh air from the ocean feels amazing. I like watching movies on weekends.
Weekends are great for enjoying films.
✅ Sample Output
yaml
Copy
Edit
Sentence 1: The ocean breeze is very refreshing.
Sentence 2: The fresh air from the ocean feels amazing.
Similarity Score: 0.82

Sentence 1: I like watching movies on weekends.
Sentence 2: Weekends are great for enjoying films.
Similarity Score: 0.85
🛠 Tech Stack
Python

Streamlit

Sentence Transformers

PDFPlumber

📌 To-Do (Future)
 Allow export of results to CSV

 Add document summarization for large PDFs

 Option to adjust similarity threshold

🙌 Author
Thejas
Connect with me on LinkedIn or feel free to fork and contribute!

vbnet
Copy
Edit

Let me know if you want to personalize the author link or repo link with your real GitHub username a
