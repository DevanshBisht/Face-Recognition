# Face-Recognition


# Face Recognition API

This Flask-based REST API performs facial recognition by comparing uploaded face images with a stored database of embeddings. It supports both 1:N (one-to-many) and 1:1 (one-to-one) face matching using cosine similarity.

## 📦 Features

- **1:N Matching**: Upload an image to find the closest match from stored candidate embeddings.
- **1:1 Matching**: Upload an image and compare it with the last accessed `_live` image.
- **Retrieve Candidate Images**: Fetch the registered and live image of a candidate using their hallticket number.
- **Cosine Similarity**: Computes similarity between embeddings to determine identity confidence.


1. POST /match — 1:N Matching (Find matching candidate)
Request:

multipart/form-data

file: Image file to match against the stored embeddings.

Response:

json
Copy
Edit
{
  "matched_hallticket": "HT123456",
  "confidence_score": 0.87,
  "image_base64": {
    "registered": "<base64_string>",
    "live": "<base64_string>"
  }
}
2. POST /match — Retrieve Candidate Images by Hallticket
Request:

application/json or form-data

hallticket: Hallticket number

Response:

json
Copy
Edit
{
  "matched_hallticket": "HT123456",
  "image_base64": {
    "registered": "<base64_string>",
    "live": "<base64_string>"
  }
}
3. POST /match — 1:1 Matching with Last Live Image
Request:

multipart/form-data

compare_file: Image to compare against the last accessed _live image

Response:

json
Copy
Edit
{
  "confidence_score": 0.92
}
🧠 How It Works
Each candidate embedding is compared against the uploaded face image using cosine distance.

The best match is selected based on the lowest distance (i.e., highest similarity).

Face alignment and embedding extraction are handled by detect_and_align_face() and extract_embedding() in Loader.py.
