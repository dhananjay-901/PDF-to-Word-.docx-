from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import uuid
from werkzeug.utils import secure_filename
import re
from typing import List, Dict

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory store for document contexts
DOCS: Dict[str, Dict] = {}
LAST_UID: str = ''

def extract_text_pypdf(path):
    try:
        import PyPDF2
        text = []
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text() or ''
                text.append(t)
        return '\n'.join(text).strip()
    except Exception:
        return ''

def extract_text_ocr(path):
    try:
        from pdf2image import convert_from_path
        import pytesseract
        pages = convert_from_path(path)
        out = []
        for img in pages:
            out.append(pytesseract.image_to_string(img))
        return '\n'.join(out).strip()
    except Exception:
        return ''

def extract_text(path):
    t = extract_text_pypdf(path)
    if len(t) >= 20:
        return t
    ocr = extract_text_ocr(path)
    return ocr if ocr else t

def write_docx(text, out_path):
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)
    for line in text.split('\n'):
        doc.add_paragraph(line)
    doc.save(out_path)

def split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not parts:
        parts = [line.strip() for line in text.split('\n') if line.strip()]
    return parts

def build_index(uid: str, text: str):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        DOCS[uid] = {'text': text, 'paragraphs': split_paragraphs(text)}
        return

    paragraphs = split_paragraphs(text)
    vec = TfidfVectorizer(stop_words='english')
    X = vec.fit_transform(paragraphs) if paragraphs else None
    DOCS[uid] = {
        'text': text,
        'paragraphs': paragraphs,
        'vectorizer': vec,
        'matrix': X
    }

def answer_query(uid: str, query: str) -> str:
    ctx = DOCS.get(uid)
    if not ctx:
        return 'No document context found. Please upload a PDF first.'
    paras = ctx.get('paragraphs', [])
    vec = ctx.get('vectorizer')
    X = ctx.get('matrix')
    if not paras:
        return 'The document contains no readable text.'
    if vec is None or X is None:
        # Fallback: keyword match scoring
        q = query.lower()
        scored = []
        for i, p in enumerate(paras):
            score = sum(word in p.lower() for word in set(re.findall(r"\w+", q)))
            scored.append((score, i))
        scored.sort(reverse=True)
        top = [paras[i] for score, i in scored[:3] if score > 0]
        return '\n\n'.join(top) if top else 'No relevant passages found.'

    try:
        qv = vec.transform([query])
        sims = (qv * X.T).A[0] if hasattr(X, 'T') else cosine_similarity(qv, X)[0]
        top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:3]
        top_paras = [paras[i] for i in top_idx if sims[i] > 0.01]
        if not top_paras:
            return 'No relevant passages found.'
        return '\n\n'.join(top_paras)
    except Exception:
        return 'Unable to process the query at this time.'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    f = request.files.get('file')
    if not f:
        return jsonify({'error': 'No file provided'}), 400
    filename = secure_filename(f.filename)
    if not filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400
    uid = uuid.uuid4().hex
    pdf_path = os.path.join(UPLOAD_DIR, uid + '_' + filename)
    f.save(pdf_path)
    text = extract_text(pdf_path)
    out_name = uid + '.docx'
    out_path = os.path.join(OUTPUT_DIR, out_name)
    write_docx(text if text else 'No text detected', out_path)
    build_index(uid, text if text else '')
    global LAST_UID
    LAST_UID = uid
    return jsonify({'download_url': '/download/' + out_name, 'uid': uid})

@app.route('/download/<name>')
def download(name):
    return send_from_directory(OUTPUT_DIR, name, as_attachment=True)

@app.route('/ask', methods=['POST'])
def ask():
    payload = request.json or {}
    q = payload.get('message', '')
    uid = payload.get('uid') or LAST_UID
    if not q:
        return jsonify({'reply': 'Please enter a question about your document.'})
    if not uid:
        return jsonify({'reply': 'No document uploaded yet. Please upload a PDF.'})
    ans = answer_query(uid, q)
    return jsonify({'reply': ans})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
