import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from pdfminer.high_level import extract_text # For PDF extraction
# Assuming your helper functions are in fact_verification.py
from fact_verification import verify_fact, detect_deepfake, evaluate_research_paper

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Set a maximum content length (e.g., 16MB)
# Clients sending files larger than this will get a 413 error
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Routes ---

# Home Route - Serves Frontend
@app.route("/")
def home():
    # Renders the main HTML page from the 'templates' folder
    return render_template("index.html")

# Fact Verification API (for text claims)
@app.route("/verify", methods=["POST"])
def fact_check():
    try:
        # Ensure request body is JSON
        data = request.get_json()
        if not data or "claim" not in data:
             return jsonify({"error": "Request must be JSON with a 'claim' field."}), 400

        claim = data.get("claim", "")
        if not claim:
            # Return 400 Bad Request if claim is empty
            return jsonify({"error": "No claim provided"}), 400

        # Call the fact verification function (from fact_verification.py)
        truth_score, explanation, sources = verify_fact(claim)

        # Return the result as JSON
        return jsonify({
            "type": "fact_check", # Add type for frontend clarity
            "truth_score": truth_score,
            "explanation": explanation,
            "sources": sources or [] # Ensure sources is always a list
        })
    except Exception as e:
        # Log the error server-side for debugging
        print(f"ERROR in /verify: {e}")
        # Return a generic 500 Internal Server Error
        return jsonify({"error": "An internal server error occurred during fact-checking."}), 500


# Deepfake Detection API (for images)
@app.route("/detect", methods=["POST"])
def deepfake_detect():
    # Check if the 'image' file part is in the request
    if "image" not in request.files:
        return jsonify({"error": "No image file part in the request"}), 400

    file = request.files["image"]

    # Check if a file was actually selected
    if file.filename == '':
        return jsonify({"error": "No file selected for upload"}), 400

    if file:
        # Secure the filename to prevent path traversal issues
        filename = secure_filename(file.filename)
        # Basic check for common image extensions
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
        _, ext = os.path.splitext(filename)
        if ext.lower() not in allowed_extensions:
             return jsonify({"error": "Invalid file type. Please upload an image (png, jpg, jpeg, gif, webp)."}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            # Save the file temporarily
            file.save(file_path)
            # Call the deepfake detection function (from fact_verification.py)
            result = detect_deepfake(file_path)

            # Check if deepfake detection itself returned an error
            if result.get("error"):
                 # Return 500 for internal processing errors
                 # Log the specific error from the function
                 print(f"Deepfake detection error for {filename}: {result.get('error')}")
                 return jsonify({"error": result.get("error")}), 500

            # Success - return the scores
            return jsonify({
                 "type": "deepfake_detection", # Add type
                 "real_score": result.get("real_score"),
                 "fake_score": result.get("fake_score")
            })
        except Exception as e:
            # Catch other potential errors (e.g., saving the file)
            print(f"Error processing image {filename}: {e}") # Log error
            return jsonify({"error": f"Could not process image: {str(e)}"}), 500
        finally:
             # Ensure the temporary file is always deleted
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as remove_error:
                     print(f"Error removing temp image file {file_path}: {remove_error}")

    # Fallback error if 'file' object exists but processing fails unexpectedly
    return jsonify({"error": "Image file processing failed unexpectedly."}), 500


# PDF Processing API (Evaluation)
@app.route("/detect_pdf", methods=["POST"])
def detect_pdf():
    # Check if the 'pdf' file part is in the request
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file part in the request"}), 400

    file = request.files["pdf"]

    # Check if a file was actually selected
    if file.filename == '':
        return jsonify({"error": "No file selected for upload"}), 400

    # Check if file has a name and ends with .pdf (case-insensitive)
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        extracted_text = None
        preview_text = ""
        try:
            # Save the file temporarily
            file.save(file_path)

            # Extract text from the PDF using pdfminer.six
            # Wrap extraction in its own try-except
            try:
                 extracted_text = extract_text(file_path)
                 if extracted_text:
                      preview_text = extracted_text[:500] + ('...' if len(extracted_text) > 500 else '')
            except Exception as extraction_error:
                 # If extraction fails, log it and return a specific error
                 print(f"Error extracting text from PDF {filename}: {extraction_error}")
                 # Return 500 as it's a server-side processing failure
                 return jsonify({"error": f"Could not extract text from PDF. It might be image-based, encrypted, or corrupted."}), 500
            finally:
                 # Always try to remove the temp file after attempting extraction
                if os.path.exists(file_path):
                     try:
                        os.remove(file_path)
                        print(f"Removed temp PDF: {file_path}")
                     except OSError as remove_error:
                         print(f"Error removing temp PDF file {file_path}: {remove_error}")

            # Check if extraction yielded any text
            if not extracted_text or not extracted_text.strip():
                 # Return OK status but indicate no text found - use evaluation type for consistency
                 return jsonify({"type": "evaluation", "preview": preview_text, "score_percent": "N/A", "justification": "No text could be extracted from the PDF."}), 200

            # --- Call the evaluation function (from fact_verification.py) ---
            score_percent, justification = evaluate_research_paper(extracted_text)

            # Check if evaluation returned an error state handled as regular output
            if score_percent in ["Error", "N/A", "Blocked"]:
                 # Return 200 OK, let frontend display the specific status/message
                 return jsonify({"type": "evaluation", "preview": preview_text, "score_percent": score_percent, "justification": justification}), 200

            # Successful evaluation case
            return jsonify({
                "type": "evaluation",
                "preview": preview_text,
                "score_percent": score_percent,
                "justification": justification
            })

        except Exception as e: # Catch unexpected errors (e.g., during file save)
            print(f"Unexpected error processing PDF {filename}: {e}")
            # Ensure cleanup happens on unexpected error too
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as remove_error:
                    print(f"Error removing temp PDF file {file_path} on unexpected error: {remove_error}")
            return jsonify({"error": f"An unexpected error occurred processing the PDF: {str(e)}"}), 500
    else:
         # If file is not a PDF based on filename
         return jsonify({"error": "Invalid file type. Please upload a PDF file."}), 400

# --- Main Execution ---
if __name__ == "__main__":
    # Use Waitress or Gunicorn for production, Flask dev server is for development
    print("Starting Flask development server...")
    # Listen on all interfaces (0.0.0.0) if you need to access it from other devices on network
    app.run(debug=True, host='0.0.0.0', port=5000) # Set debug=False for production!