# ⚖️ Legal PDF Analyzer

Legal PDF Analyzer is an interactive web application built with Streamlit that allows users to upload and analyze legal documents in PDF format. It uses Google Gemini's AI capabilities to extract insights, classify document types, and answer user questions based on the document content.

---

## 🚀 Features

### 📂 Tab 1: Document Overview
- Upload multiple legal PDF documents.
- Get a summary of:
  - Total number of documents uploaded.
  - File names and sizes.
  - First-page text preview from each document.
  - Predicted **legal category** of each document (e.g., Contract Law, Property Law, Employment Law).

### 💬 Tab 2: PDF Analyzer (Q&A)
- Ask questions directly from the uploaded documents.
- Uses **Google Generative AI (Gemini)** via LangChain to retrieve accurate, context-based answers.
- Built-in FAISS vector database for efficient similarity search on extracted text.
- Custom prompt template to ensure legally sound and accurate responses.

---

## 🧠 Powered By

- **Streamlit** – For building the interactive frontend.
- **LangChain** – For chaining prompts and managing document pipelines.
- **Google Generative AI (Gemini)** – For both embeddings and chat model.
- **FAISS** – For semantic search over document chunks.
- **PyPDF2** – For reading PDF files.

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/legal-pdf-analyzer.git
cd legal-pdf-analyzer
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🔑 API Key Setup

This app uses **Google Gemini** for text embeddings and chat responses.

To use it:
1. Visit [Google AI Studio](https://makersuite.google.com/).
2. Generate an API Key.
3. Enter your API Key in the **sidebar input field** when prompted after launching the app.

> **Note:** Your key is not stored permanently—it is only used for your current session.

---

## 🗂 Example Use Cases

- Analyze lengthy contracts to identify key clauses.
- Classify uploaded documents into different legal domains.
- Query specific points in litigation reports or real estate deeds.
- Summarize lengthy court rulings or IP documents.

---

## 📁 Project Structure

```
legal-pdf-analyzer/
├── app.py               # Main Streamlit application
├── requirements.txt     # List of dependencies
└── README.md            # Project documentation
```

---

## 📜 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software.

---

## 👨‍💻 Author

Developed by [Your Name / GitHub Handle]  
Feel free to reach out for contributions, feedback, or collaboration!
