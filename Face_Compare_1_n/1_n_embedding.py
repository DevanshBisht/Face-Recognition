import os
import csv
import numpy as np
import sys
from scipy.spatial.distance import cosine
from Loader import detect_and_align_face, extract_embedding  # Import functions

# Get base directory (works for both script & .exe)
base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

# Define file path for stored embeddings
embedding_file = os.path.join(base_dir, "hallticket_embeddings.csv")

# Load stored embeddings from CSV
stored_embeddings = {}

with open(embedding_file, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip header

    for row in reader:
        image_path = row[0]  # Full path of the image
        hallticket_number = os.path.basename(image_path)  # Extract filename (hallticket number)
        embedding_vector = np.array(row[1:], dtype=float)  # Convert features to float
        stored_embeddings[hallticket_number] = embedding_vector

# Function to compare a single live image against all stored embeddings
def compare_single_image(live_image_path):
    if not os.path.exists(live_image_path):
        print(f"Error: File not found - {live_image_path}")
        return None

    print(f"Processing live image: {live_image_path}")

    # Detect and align face
    aligned_face, face_count = detect_and_align_face(live_image_path)

    if face_count == 1:
        # Extract embedding for the live image
        live_embedding = extract_embedding(aligned_face)
        live_embedding = np.array(live_embedding).flatten()
        print(live_embedding)

        best_match = None
        best_similarity = float('inf')  # Lower cosine distance means better match

        # Compare with stored embeddings (1:N comparison)
        for hallticket_number, hall_embedding in stored_embeddings.items():
            similarity = cosine(hall_embedding, live_embedding)

            # Update best match if similarity is lower
            if similarity < best_similarity:
                best_similarity = similarity
                best_match = hallticket_number  # Store the hallticket number

        if best_match:
            confidence_score = 1 - best_similarity  # Similarity score
            print(f"Best Match: {best_match}, Confidence Score: {confidence_score:.4f}")
            return best_match, confidence_score
    else:
        print(f"No or multiple faces detected in {live_image_path}, skipping.")
        return None

# Example usage: Call function with a single image path
live_image_path = "D:\\Learn\\Finetuned_Facenet_TEST\\150_photos\\live_new_20\\TH2502TVP293AAB2131_live.jpg"
result = compare_single_image(live_image_path)

if result:
    hallticket_number, confidence_score = result
    print(f"\nFinal Result: Live Image -> {live_image_path}, Matched Hallticket -> {hallticket_number}, Confidence Score -> {confidence_score:.4f}")
