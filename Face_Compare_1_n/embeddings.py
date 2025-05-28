


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
