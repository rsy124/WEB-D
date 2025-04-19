# VeriAI - Fact Checking & Deepfake Detection Tool 🔍🤖

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey)](https://flask.palletsprojects.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/)

---

## 📌 Overview

**VeriAI** is an AI-powered platform that enables users to:
- ✅ Fact-check any input text
- 📁 Upload files for verification
- 🖼️ Detect deepfakes using image recognition models  
It uses Natural Language Processing and computer vision techniques to ensure the reliability of the content users consume or share.

---

## 🚀 Features

- ✍️ **Text Input & File Upload** for fact-checking
- 📊 **Truth Score** and **AI-Generated Score**
- 📌 **Verified Sources** Display
- 🖼️ **Deepfake Detection** via uploaded images (JPG/PNG)
- ⚡ Real-time feedback with interactive UI
- ⏳ Loading indicator while processing requests

---

## 🧠 Tech Stack

### 🔧 Frontend
- **HTML5**
- **CSS3**
- **JavaScript**

### 🔧 Backend
- **Python 3.9+**
- **Flask** – lightweight Python web framework

### 🤖 AI/ML Models
- **Hugging Face Transformers** for text analysis
- **Pre-trained Deepfake Detection Model** (image classifier)

---

## 📂 Project Structure

```bash
VeriAI/
│
├── static/
│   └── style.css          # Styling for the frontend
├── templates/
│   └── index.html         # Main UI
│
├── app.py                 # Flask backend
├── script.js              # Client-side logic (fact check, loading, etc.)
├── model.py               # Deepfake detection logic (if separate)
├── requirements.txt       # Python dependencies
└── README.md              # You're here!
```

---

## 🖥️ Screenshots

![UI Screenshot](screenshots/ui.png)
![Deepfake Detection](screenshots/deepfake.png)

---

## 🔄 How It Works

1. User inputs text or uploads a file/image.
2. Data is sent to Flask backend via API.
3. Flask uses:
   - Hugging Face model to extract named entities and cross-verify with reliable sources.
   - Image recognition model to detect manipulated content.
4. Returns:
   - Truth Score
   - AI-Generated Probability
   - Verified sources (URLs)
   - Deepfake Result (if image uploaded)

---

## 🧪 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/veriai.git
cd veriai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Flask server
python app.py
```

Access it via `http://localhost:5000` in your browser.

---

## 📈 Future Scope

- 🗣️ Voice input for fact-checking
- 🌐 Multilingual support
- 🧠 Custom AI model fine-tuning
- 📱 Android/Web App Integration

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Submit pull requests
- Report issues
- Suggest new features

---

## 📄 License

This project is licensed under the MIT License.

---

> Created with ❤️ by Team VeriAI
```
