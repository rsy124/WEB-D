
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from pdfminer.high_level import extract_text  # For PDF extraction
from fact_verification import verify_fact, detect_deepfake, evaluate_research_paper, verify_document

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST", "HEAD", "OPTIONS"])
def home():
    return render_template("index.html")

@app.route("/verify", methods=["POST"])
def fact_check():
    try:
        data = request.get_json()
        if not data or "claim" not in data:
            return jsonify({"error": "Request must be JSON with a 'claim' field."}), 400
        claim = data.get("claim", "")
        if not claim:
            return jsonify({"error": "No claim provided"}), 400
        truth_score, explanation, sources = verify_fact(claim)
        return jsonify({
            "type": "fact_check",
            "truth_score": truth_score,
            "explanation": explanation,
            "sources": sources or []
        })
    except Exception as e:
        print(f"ERROR in /verify: {e}")
        return jsonify({"error": "An internal server error occurred during fact-checking."}), 500

@app.route("/detect", methods=["POST"])
def deepfake_detect():
    if "image" not in request.files:
        return jsonify({"error": "No image file part in the request"}), 400
    file = request.files["image"]
    if file.filename == '':
        return jsonify({"error": "No file selected for upload"}), 400
    if file:
        filename = secure_filename(file.filename)
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
        _, ext = os.path.splitext(filename)
        if ext.lower() not in allowed_extensions:
            return jsonify({"error": "Invalid file type. Please upload an image (png, jpg, jpeg, gif, webp)."}), 400
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            result = detect_deepfake(file_path)
            if result.get("error"):
                print(f"Deepfake detection error for {filename}: {result.get('error')}")
                return jsonify({"error": result.get("error")}), 500
            return jsonify({
                "type": "deepfake_detection",
                "real_score": result.get("real_score"),
                "fake_score": result.get("fake_score")
            })
        except Exception as e:
            print(f"Error processing image {filename}: {e}")
            return jsonify({"error": f"Could not process image: {str(e)}"}), 500
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as remove_error:
                    print(f"Error removing temp image file {file_path}: {remove_error}")
    return jsonify({"error": "Image file processing failed unexpectedly."}), 500

@app.route("/evaluate", methods=["POST"])
def detect_pdf():
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file part in the request"}), 400
    file = request.files["pdf"]
    if file.filename == '':
        return jsonify({"error": "No file selected for upload"}), 400
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        extracted_text = None
        preview_text = ""
        try:
            file.save(file_path)
            try:
                extracted_text = extract_text(file_path)
                if extracted_text:
                    preview_text = extracted_text[:500] + ('...' if len(extracted_text) > 500 else '')
            except Exception as extraction_error:
                print(f"Error extracting text from PDF {filename}: {extraction_error}")
                return jsonify({"error": "Could not extract text from PDF. It might be image-based, encrypted, or corrupted."}), 500
            finally:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except OSError as remove_error:
                        print(f"Error removing temp PDF file {file_path}: {remove_error}")
            if not extracted_text or not extracted_text.strip():
                return jsonify({"type": "evaluation", "preview": preview_text, "score_percent": "N/A", "justification": "No text could be extracted from the PDF."}), 200
            score_percent, justification = evaluate_research_paper(extracted_text)
            return jsonify({
                "type": "evaluation",
                "preview": preview_text,
                "score_percent": score_percent,
                "justification": justification
            })
        except Exception as e:
            print(f"Unexpected error processing PDF {filename}: {e}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as remove_error:
                    print(f"Error removing temp PDF file {file_path} on unexpected error: {remove_error}")
            return jsonify({"error": f"An unexpected error occurred processing the PDF: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a PDF file."}), 400

@app.route("/verify-document", methods=["POST"])
def document_verification():
    if "document" not in request.files:
        return jsonify({"error": "No document uploaded"}), 400
    file = request.files["document"]
    if file.filename == '':
        return jsonify({"error": "No file selected for upload"}), 400
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            result = verify_document(file_path)
            return jsonify(result)
        except Exception as e:
            print(f"Error verifying document {filename}: {e}")
            return jsonify({"error": f"Error verifying document: {str(e)}"}), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        return jsonify({"error": "Invalid file type. Only PDF files are supported for verification."}), 400

if __name__ == "__main__":
    print("Starting Flask development server...")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
