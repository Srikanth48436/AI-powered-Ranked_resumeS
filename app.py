import os
import uuid
import spacy
import fitz  # PyMuPDF
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Preprocess text using spaCy
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# Index route (GET)
@app.route('/')
def index():
    return render_template('index.html')

# Upload route (POST)
@app.route('/upload', methods=['POST'])
def upload():
    # Get job description and preprocess it
    job_desc = request.form.get('job_desc', '').strip()
    if not job_desc:
        return "Please enter a job description.", 400

    job_desc_processed = preprocess_text(job_desc)

    # Get uploaded files
    files = request.files.getlist('resumes')
    if not files:
        return "No resumes uploaded.", 400

    results = []

    for file in files:
        if file and file.filename.endswith('.pdf'):
            # Make filename unique and secure
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            # Extract and preprocess resume text
            raw_text = extract_text_from_pdf(filepath)
            preprocessed_resume = preprocess_text(raw_text)

            # TF-IDF Vectorization and Similarity
            tfidf = TfidfVectorizer()
            vectors = tfidf.fit_transform([job_desc_processed, preprocessed_resume])
            score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

            results.append((file.filename, score))

    # Sort results by score (highest first)
    results.sort(key=lambda x: x[1], reverse=True)

    return render_template('results.html', results=results)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
