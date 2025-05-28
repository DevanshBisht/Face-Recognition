# import cv2
# import numpy as np
# import os
# import pandas as pd
# from deepface import DeepFace
# from Loader import extract_embedding  # Assuming `extract_embedding` is in Loader.py
#
# # Define paths
# hallticket_folder = 'D:\\Learn\\Finetuned_Facenet_TEST\\150_photos\\hallticket'
# embedding_file = 'D:\\Learn\\Finetuned_Facenet_TEST\\150_photos\\hallticket_embeddings.csv'
#
# # Get all image paths
# hallticket_images = [os.path.join(hallticket_folder, f) for f in os.listdir(hallticket_folder) if
#                      f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#
# if not hallticket_images:
#     print("No hallticket images found! Check the folder path.")
#     exit(1)
#
# print(f"🖼️ Found {len(hallticket_images)} images. Generating embeddings...")
#
# embeddings = []
#
# # Loop through each image and extract embeddings
# for img_path in hallticket_images:
#     try:
#         # Step 1: Detect and align face
#         print(f"Detecting and aligning face for {img_path}")
#
#         # Detect and align face using DeepFace's detect_face function (or use your own method)
#         aligned_face = DeepFace.detectFace(img_path, detector_backend='opencv')  # or use 'mtcnn', 'dlib', etc.
#
#         # Step 2: Extract embeddings after face alignment
#         print(f"Extracting embedding for {img_path}")
#         img_array = cv2.imread(img_path)
#
#         # Ensure that we pass the aligned face to the embedding extraction function
#         embeddings_from_image = extract_embedding(aligned_face)  # Assuming `extract_embedding` works with aligned face
#
#         if embeddings_from_image:
#             embeddings.append([img_path] + embeddings_from_image[0])  # Save the image path and embedding
#
#             print(f"✅ Embedding for {img_path} extracted.")
#         else:
#             print(f"No embedding for {img_path}")
#
#     except Exception as e:
#         print(f"Error processing {img_path}: {e}")
#
# # Save embeddings to CSV
# df = pd.DataFrame(embeddings, columns=["Image Path"] + [f"Embedding_{i + 1}" for i in range(len(embeddings[0]) - 1)])
# df.to_csv(embedding_file, index=False)
#
# print(f"✅ All embeddings saved to {embedding_file}")


# import os
# import csv
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations
# import cv2
# import numpy as np
# from pathlib import Path
# from PIL import Image
# import sys
#
#
# # Import required functions from Loader.py
# from Loader import detect_and_align_face, extract_embedding
#
# # Define the folder where images are stored and output CSV file
# hallticket_folder = 'D:\\Learn\\Finetuned_Facenet_TEST\\150_photos\\hallticket'
# embedding_file = 'D:\\Learn\\Finetuned_Facenet_TEST\\150_photos\\hallticket_embeddings.csv'
#
# # Create a list to hold the image file names and embeddings
# embeddings_data = []
#
# # Iterate through all the images in the specified folder
# for image_name in os.listdir(hallticket_folder):
#     image_path = os.path.join(hallticket_folder, image_name)
#
#     # Check if the file is an image (you can modify this check if needed)
#     if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#         print(f"Processing image: {image_name}")
#
#         # Detect and align the face using the existing function
#         aligned_face, face_count = detect_and_align_face(image_path)
#
#         # Only proceed if exactly one face is detected
#         if face_count == 1:
#             # Extract embeddings using the extract_embedding function
#             img_embedding = extract_embedding(aligned_face)
#
#             # Flatten the embedding and add the image name along with the embeddings
#             flat_embedding = np.array(img_embedding).flatten()
#             embeddings_data.append([image_name] + list(flat_embedding))  # Image name followed by the embedding
#
#         else:
#             print(f"No or multiple faces detected in {image_name}, skipping.")
#
# # Save the embeddings to CSV file
# with open(embedding_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#
#     # Write the header (Image Name + Embedding Features)
#     header = ['Image Name'] + [f'Feature_{i}' for i in range(1, len(embeddings_data[0]) if embeddings_data else 1)]
#     writer.writerow(header)
#
#     # Write the embeddings data
#     writer.writerows(embeddings_data)
#
# print(f"Embeddings extraction completed. Embeddings saved to {embedding_file}")


# embedding.py

# import os
# import csv
# import numpy as np
# from pathlib import Path
# from Loader import detect_and_align_face, extract_embedding  # Import functions from Loader.py
#
#
# def compute_and_store_embeddings(image_folder: str, output_csv: str):
#     """
#     Computes and stores face embeddings for all images in the specified folder.
#     :param image_folder: Path to the folder containing images
#     :param output_csv: Path to the output CSV file to store embeddings
#     """
#
#     # Create or overwrite the CSV file
#     with open(output_csv, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Image Path', 'Embedding'])  # Header row
#
#         # Iterate through each image in the folder
#         for image_path in Path(image_folder).rglob('*'):
#             if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:  # Process only image files
#                 print(f"Processing image: {image_path}")
#
#                 # Detect and align face in the image
#                 aligned_face, face_count = detect_and_align_face(str(image_path))
#
#                 if face_count == 1:  # If exactly one face is detected
#                     # Extract embedding for the aligned face
#                     embedding = extract_embedding(aligned_face)
#
#                     # Write the image path and embedding to the CSV
#                     writer.writerow([str(image_path), ','.join(map(str, embedding))])
#                 else:
#                     print(f"Skipping {image_path}: No or multiple faces detected.")
#
#     print(f"Embeddings have been saved to {output_csv}")
#
#
# # Run the function to compute and store embeddings
# if __name__ == "__main__":
#     input_image_folder = 'D:\\Learn\\Finetuned_Facenet_TEST\\150_photos\\halticket'  # Folder with images
#     output_csv_file = 'D:\\Learn\\Finetuned_Facenet_TEST\\150_photos\\halticket_embeddings.csv'  # Output CSV file
#
#     compute_and_store_embeddings(input_image_folder, output_csv_file)

#
# import csv
# import numpy as np
# import pandas as pd
# from Loader import detect_and_align_face, extract_embedding
# import os
#
#
# def save_embedding_to_csv(image_paths, csv_filename="D:\Learn\Finetuned_Facenet_TEST\Face_Compare_1_n\embeddings_office.csv"):
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
#         if face_count == 1:
#             embedding = extract_embedding(aligned_face)
#
#             if embedding and isinstance(embedding, list) and len(embedding) > 0:
#                 embeddings_data.append([image_name, embedding])
#         else:
#             print(f"Skipping {image_name}: Face not detected correctly.")
#
#     # Ensure we have at least one valid embedding before writing to CSV
#     if embeddings_data:
#         with open(csv_filename, mode='w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(["image_name", "embedding"])
#             for data in embeddings_data:
#                 writer.writerow([data[0], data[1]])
#
#         print(f"Embeddings saved to {csv_filename}")
#     else:
#         print("No valid embeddings found. CSV file was not created.")
#
#
# # Example usage
# if __name__ == "__main__":
#     image_folder = "D:\Learn\Finetuned_Facenet_TEST\Face_Compare_1_n\HallTicketPhotosLive_appcred/21684.jpeg"  # Change this path accordingly
#     image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if
#                    img.endswith((".jpg", ".png"))]
#     save_embedding_to_csv(image_paths)


import os
import pandas as pd
from Loader import detect_and_align_face, extract_embedding

# Define your input folder path
image_folder = "D:\Learn\Finetuned_Facenet_TEST\Face_Compare_1_n\HallTicketPhotosLive_appcred"
output_csv = "embedding_report.csv"

# Initialize results list
results = []

# Loop through all image files
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)

        # Call detect_and_align_face
        aligned_img, face_count = detect_and_align_face(image_path)

        if aligned_img is None:
            if face_count == 0:
                reason = "No face detected"
            elif face_count > 1:
                reason = f"Multiple faces detected ({face_count})"
            else:
                reason = "Face alignment failed"
            status = "fail"
        else:
            # Call extract_embedding
            embeddings = extract_embedding(aligned_img)
            if embeddings and isinstance(embeddings[0], (list, tuple, np.ndarray)):
                status = "success"
                reason = "Embedding generated"
            else:
                status = "fail"
                reason = "Embedding not generated"

        results.append({
            "image_name": filename,
            "status": status,
            "reason": reason
        })

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

print(f"Embedding report saved to {output_csv}")
