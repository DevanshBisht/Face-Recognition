import os
import csv
import glob
import base64
import numpy as np
import zipfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.spatial.distance import cosine
from Loader import detect_and_align_face, extract_embedding  # Custom functions
import os
import csv
import glob
import base64
import numpy as np
import zipfile
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.spatial.distance import cosine
from Loader import detect_and_align_face, extract_embedding  # Custom functions
import cv2
import shutil


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
#CORS(app)

# Directory setup
temp_dir = os.path.join(os.getenv("TEMP"), "decrypted_files")
downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
base_dir = temp_dir

embedding_file = os.path.join(base_dir, "hallticket_embeddings.csv")
candidate_photo_dir = os.path.join(base_dir, "CandidatePhotographs")
os.makedirs(candidate_photo_dir, exist_ok=True)

# Load stored embeddings


# Directory setup
temp_dir = os.path.join(os.getenv("TEMP"), "decrypted_files")
base_dir = temp_dir

# embedding_file = os.path.join(base_dir, "hallticket_embeddings.csv")
# candidate_photo_dir = os.path.join(base_dir, "CandidatePhotographs")
# os.makedirs(candidate_photo_dir, exist_ok=True)

## Load stored embeddings
# stored_embeddings = {}
#
# # Auto-detect CSV delimiter and load embeddings
# with open(embedding_file, 'r', encoding='utf-8') as file:
#     sample = file.readline()
#     delimiter = '\t' if '\t' in sample else ','
#
# with open(embedding_file, 'r', encoding='utf-8') as file:
#     reader = csv.reader(file, delimiter=delimiter)
#     header = next(reader, None)
#
#     for row in reader:
#         if len(row) < 2:
#             continue
#         applicant_cred_id = row[0].strip()
#         try:
#             embedding_str = row[1].strip().replace("{", "").replace("}", "")
#             embedding_vector = np.array([float(x) for x in embedding_str.split(",")])
#             stored_embeddings[applicant_cred_id] = embedding_vector
#         except (ValueError, IndexError) as e:
#             print(f"Skipping row due to error: {row} -> {e}")
#
# print(f"Loaded {len(stored_embeddings)} embeddings successfully!")

# Load stored embeddings (safely)
stored_embeddings = {}

if os.path.exists(embedding_file):
    try:
        with open(embedding_file, 'r', encoding='utf-8') as file:
            sample = file.readline()
            delimiter = '\t' if '\t' in sample else ','

        with open(embedding_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=delimiter)
            header = next(reader, None)

            for row in reader:
                if len(row) < 2:
                    continue
                applicant_cred_id = row[0].strip()
                try:
                    embedding_str = row[1].strip().replace("{", "").replace("}", "")
                    embedding_vector = np.array([float(x) for x in embedding_str.split(",")])
                    stored_embeddings[applicant_cred_id] = embedding_vector
                except (ValueError, IndexError) as e:
                    print(f"Skipping row due to error: {row} -> {e}")
        print(f"Loaded {len(stored_embeddings)} embeddings successfully!")
    except Exception as e:
        print(f"Failed to load embeddings: {str(e)}")
else:
    print(f"Embedding file not found at startup: {embedding_file}")




# Store last accessed _live image path
last_live_image_path = None
# Function to load embeddings
def load_embeddings_from_temp():
    embedding_file = os.path.join(temp_dir, 'hallticket_embeddings.csv')

    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"Embedding file not found in {temp_dir}")

    stored_embeddings = {}
    with open(embedding_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',')
        header = next(reader, None)

        for row in reader:
            if len(row) < 2:
                continue
            applicant_cred_id = row[0].strip()
            try:
                embedding_str = row[1].strip().replace("{", "").replace("}", "")
                embedding_vector = np.array([float(x) for x in embedding_str.split(",")])
                stored_embeddings[applicant_cred_id] = embedding_vector
            except (ValueError, IndexError) as e:
                print(f"Skipping row due to error: {row} -> {e}")

    print(f"Loaded {len(stored_embeddings)} embeddings successfully!")
    return stored_embeddings


# Store last accessed _live image path
last_live_image_path = None

# Helpers
def encode_image_to_base64(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    return None

def find_image_path(pattern):
    matched = glob.glob(os.path.join(candidate_photo_dir, pattern))
    return matched[0] if matched else None

def compare_single_image(live_image_path):
    if not os.path.exists(live_image_path):
        return None, "Uploaded image not found"

    aligned_face, face_count = detect_and_align_face(live_image_path)
    if face_count == 0:
        return None, "No face detected in the image"
    elif face_count > 1:
        return None, "More than one face detected in the image"

    live_embedding = extract_embedding(aligned_face)
    live_embedding = np.array(live_embedding).flatten()

    best_match = None
    best_similarity = float('inf')

    for hallticket_number, hall_embedding in stored_embeddings.items():
        similarity = cosine(hall_embedding, live_embedding)
        if similarity < best_similarity:
            best_similarity = similarity
            best_match = hallticket_number

    if best_match:
        confidence_score = 1 - best_similarity
        return best_match, round(confidence_score, 4)

    return None, None


def compare_with_live_reference(uploaded_path, reference_path):
    try:
        aligned_face, face_count = detect_and_align_face(uploaded_path)
        aligned_ref, ref_count = detect_and_align_face(reference_path)
    except Exception as e:
        return None, f"Face detection failed: {str(e)}"

    if face_count == 0:
        return None, "No face detected in the uploaded image"
    elif face_count > 1:
        return None, "More than one face detected in the uploaded image"

    if ref_count == 0:
        return None, "No face detected in the reference image"
    elif ref_count > 1:
        return None, "More than one face detected in the reference image"

    upload_embedding = extract_embedding(aligned_face)
    ref_embedding = extract_embedding(aligned_ref)

    upload_embedding = np.array(upload_embedding).flatten()
    ref_embedding = np.array(ref_embedding).flatten()

    similarity = cosine(upload_embedding, ref_embedding)
    confidence_score = 1 - similarity
    return round(confidence_score, 4), None

@app.route('/match', methods=['POST'])
def match_face():
    global last_live_image_path

    # Scenario 1: Full image upload for 1:N match
    if 'file' in request.files and not (request.form or request.is_json):
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        extension = os.path.splitext(file.filename)[1]
        save_path = os.path.join(candidate_photo_dir, f"uploaded_image{extension}")
        file.save(save_path)

        hallticket_number, result = compare_single_image(save_path)

        if hallticket_number is None and isinstance(result, str):
            return jsonify({"error": result}), 400

        confidence_score = result

        candidate_image_path = find_image_path(f"{hallticket_number}.*")
        live_image_path = find_image_path(f"{hallticket_number}_live.*")

        candidate_image_base64 = encode_image_to_base64(candidate_image_path) if candidate_image_path else None
        live_image_base64 = encode_image_to_base64(live_image_path) if live_image_path else None

        response = {
            "matched_hallticket": hallticket_number,
            "confidence_score": confidence_score,
            "image_base64": {}
        }

        if candidate_image_base64:
            response["image_base64"]["registered"] = candidate_image_base64
        if live_image_base64:
            response["image_base64"]["live"] = live_image_base64

        if not candidate_image_base64 or not live_image_base64:
            missing = []
            if not candidate_image_base64:
                missing.append("registered")
            if not live_image_base64:
                missing.append("live")
            response["error"] = f"Missing image(s): {', '.join(missing)}"
            return jsonify(response), 404

        return jsonify(response)

    # Scenario 2: Hallticket provided to get images
    elif request.is_json and 'hallticket' in request.json:
        hallticket_number = request.json['hallticket'].strip()
    elif 'hallticket' in request.form:
        hallticket_number = request.form['hallticket'].strip()
    else:
        hallticket_number = None

    if hallticket_number:
        candidate_image_path = find_image_path(f"{hallticket_number}.*")
        live_image_path = find_image_path(f"{hallticket_number}_live.*")
        last_live_image_path = live_image_path  # Save for scenario 3

        candidate_image_base64 = encode_image_to_base64(candidate_image_path) if candidate_image_path else None
        live_image_base64 = encode_image_to_base64(live_image_path) if live_image_path else None

        if candidate_image_base64 or live_image_base64:
            return jsonify({
                "matched_hallticket": hallticket_number,
                "image_base64": {
                    "registered": candidate_image_base64,
                    "live": live_image_base64
                }
            })
        else:
            return jsonify({
                "error": f"No images found for hallticket {hallticket_number}"
            }), 404

    # Scenario 3: File uploaded without hallticket (1:1 with previous _live)
    if 'compare_file' in request.files:
        if not last_live_image_path:
            return jsonify({"error": "No previous hallticket reference found. Please retrieve images first."}), 400

        file = request.files['compare_file']
        extension = os.path.splitext(file.filename)[1]
        save_path = os.path.join(candidate_photo_dir, f"uploaded_temp{extension}")
        file.save(save_path)

        confidence_score, error = compare_with_live_reference(save_path, last_live_image_path)
        if error:
            return jsonify({"error": error}), 400

        return jsonify({"confidence_score": confidence_score})

    return jsonify({"error": "Invalid request"}), 400

# @app.route('/check_file', methods=['POST'])

# def check_file():
#     filename = request.form.get('file', '').strip()
#
#     if not filename:
#         return jsonify({"status": "error", "message": "Filename not provided"}), 400
#
#     file_path = os.path.join(downloads_dir, filename)  # Check file in Downloads
#
#     if not os.path.exists(file_path):
#         return jsonify({"status": "not_found", "message": f"{filename} does not exist"}), 404
#
#     if file_path.endswith('.crdownload'):
#         return jsonify({"status": "incomplete", "message": "File download is not complete"}), 409
#
#     try:
#         # Extract to Downloads directory
#         extracted_folder = os.path.join(downloads_dir, Path(filename).stem)
#
#         # Extract ZIP file
#         with zipfile.ZipFile(file_path, 'r') as zip_ref:
#             zip_ref.extractall(extracted_folder)
#
#         move_bat_path = os.path.join(extracted_folder, "move.bat")
#         print(f"move.bat path: {move_bat_path}")  # Debugging to verify the path
#
#         if os.path.exists(move_bat_path):
#             # Execute move.bat in the extracted folder
#             result = subprocess.run(
#                 ["cmd", "/c", move_bat_path],
#                 shell=True,
#                 capture_output=True,
#                 text=True,
#                 cwd=extracted_folder  # Ensure the batch file is executed in the correct directory
#             )
#
#             if result.returncode != 0:
#                 return jsonify({
#                     "status": "error",
#                     "message": "move.bat execution failed",
#                     "stderr": result.stderr.strip()
#                 }), 500
#
#             # After move.bat execution, check for the embedding file again in temp directory
#             try:
#                 stored_embeddings = load_embeddings_from_temp()
#             except FileNotFoundError as e:
#                 return jsonify({
#                     "status": "error",
#                     "message": str(e)
#                 }), 404
#
#             # Clean up the extracted folder
#             try:
#                 shutil.rmtree(extracted_folder)
#             except Exception as e:
#                 return jsonify({
#                     "status": "error",
#                     "message": f"Failed to delete extracted folder: {str(e)}"
#                 }), 500
#
#             return jsonify({
#                 "status": "success",
#                 "message": "File extracted, move.bat executed, embedding file loaded, and folder deleted.",
#                 "stdout": result.stdout.strip()
#             }), 200
#
#         else:
#             return jsonify({"status": "error", "message": "move.bat not found in extracted folder"}), 500
#
#     except zipfile.BadZipFile:
#         return jsonify({"status": "error", "message": "Provided file is not a valid ZIP"}), 400
#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/check_file', methods=['POST'])
def check_file():
    global stored_embeddings, last_live_image_path  # Declare globals to modify them here

    # Clear previous cache immediately
    stored_embeddings.clear()
    new_embeddings = load_embeddings_from_temp()
    stored_embeddings.update(new_embeddings)
    last_live_image_path = None

    filename = request.form.get('file', '').strip()

    if not filename:
        return jsonify({"status": "error", "message": "Filename not provided"}), 400

    file_path = os.path.join(downloads_dir, filename)  # Check file in Downloads

    if not os.path.exists(file_path):
        return jsonify({"status": "not_found", "message": f"{filename} does not exist"}), 404

    if file_path.endswith('.crdownload'):
        return jsonify({"status": "incomplete", "message": "File download is not complete"}), 409

    try:
        extracted_folder = os.path.join(downloads_dir, Path(filename).stem)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder)

        move_bat_path = os.path.join(extracted_folder, "move.bat")
        print(f"move.bat path: {move_bat_path}")

        if os.path.exists(move_bat_path):
            result = subprocess.run(
                ["cmd", "/c", move_bat_path],
                shell=True,
                capture_output=True,
                text=True,
                cwd=extracted_folder
            )

            if result.returncode != 0:
                return jsonify({
                    "status": "error",
                    "message": "move.bat execution failed",
                    "stderr": result.stderr.strip()
                }), 500

            # Reload embeddings after clearing cache

            try:
                stored_embeddings.clear()
                stored_embeddings.update(load_embeddings_from_temp())
            except FileNotFoundError as e:
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 404

            try:
                shutil.rmtree(extracted_folder)
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": f"Failed to delete extracted folder: {str(e)}"
                }), 500

            return jsonify({
                "status": "success",
                "message": "File extracted, move.bat executed, embedding file loaded, and folder deleted.",
                "stdout": result.stdout.strip()
            }), 200

        else:
            return jsonify({"status": "error", "message": "move.bat not found in extracted folder"}), 500

    except zipfile.BadZipFile:
        return jsonify({"status": "error", "message": "Provided file is not a valid ZIP"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
