from flask import Flask, render_template, request, jsonify
import os
import re
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import google.generativeai as genai
from dotenv import load_dotenv

# Download NLTK resources if not already present
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
assert API_KEY, "ERROR: Gemini API Key is missing"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro-latest") 

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize stopwords, lemmatizer, and unnecessary words
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
unnecessary_words = {"etc", "e.t.c", "eg", "i.e", "viz"}

def clean_text(text):
    text = text.lower() 
    text = re.sub(r"[^a-zA-Z0-9.%\s]", "", text)
    words = word_tokenize(text) 
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word not in unnecessary_words]
    return " ".join(words)

def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            pdf_text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return pdf_text

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        pdf_text = extract_text_from_pdf(file_path)
        cleaned_text = clean_text(pdf_text)

        if cleaned_text:
            prompt = f"""
            Please provide a clear and concise summary of the following document:
            {cleaned_text[:30000]} 
            """
            response = model.generate_content(prompt)
            summary = response.text
            return jsonify({'summary': summary})
        else:
            return jsonify({'error': 'No text extracted from PDF'}), 500

@app.route('/generate_policy', methods=['POST'])
def generate_policy():
    policy_link = request.form.get('policy_link')
    scenario = request.form.get('scenario')

    if not policy_link or not scenario:
        return jsonify({'error': 'Policy URL and scenario are required'}), 400

    prompt = f"Generate an economic policy drawing from {policy_link} for the following scenario: {scenario}"
    response = model.generate_content(prompt)
    policy = response.text
    return jsonify({'policy': policy})

if __name__ == '__main__':
    app.run(debug=True)
