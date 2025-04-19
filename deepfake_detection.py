from flask import Flask, request, render_template, jsonify
import google.generativeai as genai
import torch
from transformers import AutoModelForImageClassification, AutoProcessor
from PIL import Image
import os

app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def verify_fact(claim):
    """Verify a fact using Gemini AI."""
    prompt = f"""
    Analyze the following claim and determine if it is true or false. Provide a truth score (0-100),
    a detailed explanation, and relevant sources.
    
    Claim: "{claim}"
    
    Return in this format:
    Truth Score: [Score]
    Explanation: [Detailed explanation]
    Sources: [List of sources]
    """
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        return result_text
    except Exception as e:
        return f"Error: {str(e)}"

# Load Deepfake Detection Model
model_name = "Hemg/Deepfake-image"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

def detect_deepfake(image_path):
    """Detect if an image is a deepfake."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    scores = predictions[0].tolist()
    return {"real": scores[1], "fake": scores[0]}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify_fact', methods=['POST'])
def verify_fact_api():
    data = request.json
    claim = data.get('claim', '')
    result = verify_fact(claim)
    return jsonify({"result": result})

@app.route('/detect_deepfake', methods=['POST'])
def detect_deepfake_api():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})
    
    image = request.files['image']
    image_path = "temp_image.jpg"
    image.save(image_path)
    
    result = detect_deepfake(image_path)
    os.remove(image_path)  # Remove temporary file
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
