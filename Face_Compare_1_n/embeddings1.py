# import os
# import csv
# import numpy as np
# from Loader import detect_and_align_face, extract_embedding  # Import functions from Loader.py
# import cv2
#
# def save_embedding_to_csv(image_paths, csv_filename="embeddings_office.csv"):
#     """
#     Extracts embeddings for given images and saves them in a CSV file.
#     :param image_paths: List of image file paths.
#     :param csv_filename: Name of the CSV file to store embeddings.
#     """
#     embeddings_data = []
#
#     for image_path in image_paths:
#         image_name = os.path.basename(image_path)
#         aligned_face, face_count = detect_and_align_face(image_path)
#
#         if face_count == 1:  # Ensure that only one face is detected
#             embedding = extract_embedding(aligned_face)
#
#             if embedding and isinstance(embedding, list) and len(embedding) > 0:
#                 embeddings_data.append([image_name] + embedding[0])  # Directly append the embedding
#         else:
#             print(f"Skipping {image_name}: Face not detected correctly or multiple faces detected.")
#
#     # Ensure we have at least one valid embedding before writing to CSV
#     if embeddings_data:
#         with open(csv_filename, mode='w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(["image_name"] + [f"embedding_{i+1}" for i in range(128)])  # 128 for 512d embeddings
#             for data in embeddings_data:
#                 writer.writerow(data)
#
#         print(f"Embeddings saved to {csv_filename}")
#     else:
#         print("No valid embeddings found. CSV file was not created.")
#
# # Example usage
# if __name__ == "__main__":
#     image_folder = "D:/Learn/Finetuned_Facenet_TEST/photo_live"  # Change this path accordingly
#     image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith((".jpg", ".png"))]
#     save_embedding_to_csv(image_paths)


import os
import psycopg2
from pathlib import Path
from Loader import detect_and_align_face, extract_embedding  # Import the functions from Loader.py
import cv2

# PostgreSQL connection details
DB_HOST = "oledbserver.pune.cdac.in"
DB_NAME = "agniveerphase7_bio"
DB_USER = "root"
DB_PASSWORD = "Password@1234#"
DB_PORT = "5434"  # Default PostgreSQL port

# Connect to PostgreSQL
def connect_db():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# Insert the embeddings into PostgreSQL
def save_embedding_to_db(hallticket_number, embedding):
    conn = connect_db()
    if conn is None:
        return

    try:
        cursor = conn.cursor()

        # Insert or update embedding into PostgreSQL table
        cursor.execute("""
            INSERT INTO applicant_image_embeddings (applicant_cred_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (applicant_cred_id) DO UPDATE 
            SET embedding = EXCLUDED.embedding;
        """, (hallticket_number, embedding))

        conn.commit()
        print(f"Embedding for {hallticket_number} saved to database.")
    except Exception as e:
        print(f"Error saving embedding for {hallticket_number}: {e}")
    finally:
        cursor.close()
        conn.close()

# Function to generate and store the embedding for a single image
def generate_and_store_embedding(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image {image_path}. Skipping.")
        return

    image = image[..., ::-1]  # Convert BGR to RGB if necessary

    # Detect and align the face in the image
    image_aligned, face_count = detect_and_align_face(image_path)

    if face_count == 1:
        # Extract the embedding
        embedding = extract_embedding(image_aligned)[0]  # Get the first embedding

        # Check if embedding is already a list; if it's a NumPy array, convert to list
        embedding_list = embedding.tolist() if not isinstance(embedding, list) else embedding

        # Extract hall ticket number (remove '_live' from filename)
        image_name = Path(image_path).stem  # Get filename without extension
        hallticket_number = image_name.replace("_live", "")  # Remove "_live"

        # Save the embedding to PostgreSQL
        save_embedding_to_db(hallticket_number, embedding_list)
    else:
        print(f"Skipping {image_path}. Face not detected or multiple faces found.")

# Function to process all images in a folder and save their embeddings
def process_images_in_folder(image_folder):
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if
                   img.endswith((".jpg", ".png"))]
    for image_path in image_paths:
        generate_and_store_embedding(image_path)

if __name__ == "__main__":
    image_folder = "D:\Learn\Finetuned_Facenet_TEST\Face_Compare_1_n\HallTicketPhotosLive_appcred"  # Path to your image folder
    process_images_in_folder(image_folder)