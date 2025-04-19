# fact_verification.py

import os
import re
from dotenv import load_dotenv
import google.generativeai as genai # Main import
# Note: genai.types will be used directly below, no separate import needed if genai is imported
import torch
from transformers import AutoModelForImageClassification, AutoProcessor
from PIL import Image, UnidentifiedImageError

# --- Add Warning Suppression ---
import warnings
try:
    # Suppress the specific FutureWarning from huggingface_hub if possible
    from huggingface_hub.file_download import CCompletionProgress # Check if class exists
    warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')
    print("Suppressed specific huggingface_hub FutureWarning.")
except ImportError:
    # Fallback to suppressing all FutureWarnings if specific import fails
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("Suppressed general FutureWarning (huggingface_hub specific import failed).")
# -----------------------------

# --- Environment Setup ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm_model = None # Initialize

if not GEMINI_API_KEY:
    print("CRITICAL: GEMINI_API_KEY environment variable not set. AI features will be unavailable.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Use latest Flash model - potentially make this configurable
        MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
        print(f"Configuring Gemini AI Model: {MODEL_NAME}...")
        llm_model = genai.GenerativeModel(MODEL_NAME)
        # Consider a quick test call here to ensure the API key is valid if needed
        print(f"Gemini AI Model configured successfully.")
    except Exception as e:
        print(f"FATAL: Error configuring Gemini AI with key: {e}")
        # Exiting or raising might be appropriate if LLM is critical
        # raise RuntimeError(f"Failed to configure Gemini AI: {e}") from e

# --- Deepfake Detection Model Setup ---
deepfake_processor = None
deepfake_model = None
try:
    # Use environment variable for model name or default
    DEEPFAKE_MODEL_NAME = os.getenv("DEEPFAKE_MODEL", "Hemg/Deepfake-image")
    print(f"Loading Deepfake model: {DEEPFAKE_MODEL_NAME}...")
    deepfake_processor = AutoProcessor.from_pretrained(DEEPFAKE_MODEL_NAME)
    deepfake_model = AutoModelForImageClassification.from_pretrained(DEEPFAKE_MODEL_NAME)
    print("Deepfake Detection Model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load Deepfake model '{DEEPFAKE_MODEL_NAME}'. Deepfake detection will be unavailable. Error: {e}")

# --- Function Definitions ---

def verify_fact(claim):
    """
    Verify a simple text claim using the configured LLM.
    Returns: truth_score (str), explanation (str), sources (list)
    """
    if not llm_model:
        return "N/A", "LLM model is not configured or failed to load.", []

    prompt = f"""
    Analyze the following claim and determine if it is true, false, or uncertain based on current, verifiable knowledge up to your last update.
    Claim: "{claim}"

    Format your response strictly as follows, with each item on a new line:
    Truth Score: [Provide a numerical score from 0 (Definitely False) to 100 (Definitely True). Use 50 for Uncertain/Cannot Verify/Opinion.]
    Explanation: [Provide a concise explanation for the score, mentioning key evidence or lack thereof. State if it's opinion-based.]
    Sources: [List up to 2 relevant, highly credible source URLs (like primary sources or reputable encyclopedias) if verifiable and applicable. If none, write "None".]
    """
    try:
        # Define safety settings to block harmful content if necessary
        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     # Add other categories as needed
        # ]

        # >>> CHANGE: Apply low temperature for potentially more consistent fact-checking too (optional) <<<
        generation_config_factcheck = genai.types.GenerationConfig(
            temperature=0.2 # Slightly higher than eval, but still low-ish
        )

        response = llm_model.generate_content(
            prompt,
            generation_config=generation_config_factcheck
            #, safety_settings=safety_settings # Uncomment safety_settings if using them
        )

        # Enhanced response checking
        if not response.candidates:
            try:
                block_reason = response.prompt_feedback.block_reason
                print(f"Fact-check prompt blocked. Reason: {block_reason}")
                # Return specific "Blocked" status
                return "Blocked", f"Content blocked by safety filter ({block_reason})", []
            except (AttributeError, ValueError, Exception):
                # Handle cases where feedback isn't available or structured as expected
                print("Fact-check response empty or invalid, no candidates.")
                return "N/A", "No valid response generated by the AI model.", []
        if not response.text:
                print("Fact-check response text is empty.")
                return "N/A", "Empty response text from the AI model.", []

        # Parsing the response (more robustly)
        result_text = response.text.strip()
        truth_score = "50" # Default to uncertain
        explanation = "Could not parse explanation from AI response."
        sources = []

        # Use MULTILINE flag for patterns starting lines
        score_match = re.search(r"^Truth Score:\s*(\d+)", result_text, re.IGNORECASE | re.MULTILINE)
        if score_match:
            truth_score = score_match.group(1)

        exp_match = re.search(r"^Explanation:\s*(.*?)(?=^\s*Sources:|$)", result_text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if exp_match:
            explanation = exp_match.group(1).strip()

        src_match = re.search(r"^Sources:\s*(.*)", result_text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if src_match:
            src_line = src_match.group(1).strip()
            if src_line.lower() not in ['none', 'n/a', '']:
                # Extract URLs, attempt to handle variations
                potential_urls = re.findall(r'https?://[^\s,"\'<>]+', src_line)
                sources = [url.strip().rstrip('.,') for url in potential_urls if url.strip()] # Clean trailing chars

        return truth_score, explanation, sources

    except Exception as e:
        print(f"Error during Gemini API call in verify_fact: {e}")
        error_details = str(e)
        # Check for specific API errors if the library provides them (e.g., API key issues, quota)
        # Example: if 'API key not valid' in error_details: ...
        return "Error", f"An API error occurred: {error_details}", []


def detect_deepfake(image_path):
    """
    Detect if an image is a deepfake using the loaded model.
    Returns: dict with 'real_score' and 'fake_score' (float percentages) or 'error'.
    """
    if not deepfake_model or not deepfake_processor:
        print("Deepfake model not loaded, cannot perform detection.")
        return {"error": "Deepfake model is not available."}

    try:
        # Ensure image path exists before opening
        if not os.path.exists(image_path):
                print(f"Error: Image file not found at path: {image_path}")
                return {"error": "Image file not found on server."}

        image = Image.open(image_path).convert("RGB")
        inputs = deepfake_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = deepfake_model(**inputs)

        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        scores = predictions[0].tolist() # Assumes model output: [fake, real] - MUST BE CONFIRMED

        if len(scores) < 2:
                print(f"Error: Unexpected model output scores for {image_path}: {scores}")
                return {"error": "Invalid model output format for deepfake scores."}

        # Ensure scores are non-negative before division
        safe_scores = [max(0, s) for s in scores]
        total = sum(safe_scores)

        fake_score = (safe_scores[0] / total) * 100 if total > 0 else 0
        real_score = (safe_scores[1] / total) * 100 if total > 0 else 0

        return {
            "fake_score": round(fake_score, 2),
            "real_score": round(real_score, 2)
        }
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file: {image_path}")
        return {"error": "Cannot identify image file. It might be corrupted or not a supported format."}
    except FileNotFoundError: # Should be caught by os.path.exists, but as fallback
         print(f"Error: Image file not found at path: {image_path}")
         return {"error": "Image file could not be accessed on server."}
    except Exception as e:
        print(f"Error during deepfake detection for {image_path}: {e}")
        return {"error": f"An unexpected error occurred during deepfake detection: {str(e)}"}


def evaluate_research_paper(text):
    """
    Evaluate a research paper's text using the configured LLM.
    Returns: score_percent (str/int), justification (str)
    """
    if not llm_model:
        return "N/A", "LLM model is not configured or failed to load."

    # --- Simplified Text Handling - Truncation ---
    # WARNING: This is suboptimal for long papers. Production needs better chunking/summarization.
    MAX_INPUT_CHARS = 30000 # Adjust based on model token limits and API performance
    truncated = False
    if len(text) > MAX_INPUT_CHARS:
        print(f"Warning: Input text length ({len(text)}) > {MAX_INPUT_CHARS}. Truncating for evaluation.")
        text_to_process = text[:MAX_INPUT_CHARS] # Simple truncation
        truncated = True
    else:
        text_to_process = text

    # --- Evaluation Prompt ---
    prompt = f"""
    Act as an impartial academic reviewer. Evaluate the quality of the following research paper text based *only* on the provided text and these criteria:
    1. Clarity of Research Question/Purpose: Is the main goal clearly stated and understandable?
    2. Soundness of Methodology: Are the methods described adequately and appropriate for the question?
    3. Significance & Validity of Findings/Conclusions: Are results clearly presented? Do they address the question? Are conclusions justified by the results? Is the significance discussed?
    4. Overall Structure & Clarity of Writing: Is the paper well-organized, logical, and easy to read?

    Provide your response STRICTLY in the following format:

    Score Percent: [Assign an overall quality score percentage from 0% to 100% based on the criteria. Be consistent.]
    Justification: [Provide a detailed justification explaining the score. Clearly separate strengths and weaknesses. Start with 'Strengths:' list positive aspects related to the criteria. Then start with 'Weaknesses:' list negative aspects or limitations related to the criteria. Explain how these factors combine to justify the specific percentage score assigned.]{' Note: Evaluation based on truncated text.' if truncated else ''}

    --- START RESEARCH PAPER TEXT ---
    {text_to_process}
    --- END RESEARCH PAPER TEXT ---
    """

    try:
        # >>> CHANGE: Define generation configuration with low temperature <<<
        generation_config = genai.types.GenerationConfig(
            temperature=0.1 # Set low temperature (e.g., 0.1 or 0.0) for reduced randomness
        )

        # >>> CHANGE: Pass generation_config to the API call <<<
        response = llm_model.generate_content(
            prompt,
            generation_config=generation_config # Apply the configuration
            # Add safety_settings if needed, separated by comma
            # , safety_settings=safety_settings
            )

        # Check response validity and safety feedback
        if not response.candidates:
            try:
                block_reason = response.prompt_feedback.block_reason
                print(f"Evaluation prompt blocked. Reason: {block_reason}")
                return "Blocked", f"Content blocked by safety filter ({block_reason})"
            except (AttributeError, ValueError, Exception):
                print("Evaluation response empty or invalid, no candidates.")
                return "N/A", "No valid response generated by the AI model for evaluation."
        if not response.text:
                print("Evaluation response text is empty.")
                return "N/A", "Empty response text from the AI model for evaluation."

        # Parsing the response
        result_text = response.text.strip()
        score_percent = "N/A"
        justification = "Could not parse justification from AI response."

        # Use MULTILINE flag for patterns starting lines
        score_match = re.search(r"^Score Percent:\s*(\d{1,3})\s*%?", result_text, re.IGNORECASE | re.MULTILINE)
        if score_match:
            score_value = int(score_match.group(1))
            score_percent = max(0, min(100, score_value)) # Clamp score 0-100

        # Capture justification more robustly
        just_match = re.search(r"^Justification:\s*(.*)", result_text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if just_match:
            justification = just_match.group(1).strip()
            # Add truncation note if applicable and not already added by LLM
            if truncated and "[Text Truncated]" not in justification and "truncated text" not in justification.lower():
                    justification += "\n\n(Note: This evaluation was based on truncated text due to length limitations.)"

        return score_percent, justification

    except Exception as e:
        print(f"Error during Gemini API call in evaluate_research_paper: {e}")
        return "Error", f"An API error occurred during paper evaluation: {str(e)}"