# Face-Recognition


# Face Recognition API

This Flask-based REST API performs facial recognition by comparing uploaded face images with a stored database of embeddings. It supports both 1:N (one-to-many) and 1:1 (one-to-one) face matching using cosine similarity.

## 📦 Features

- **1:N Matching**: Upload an image to find the closest match from stored candidate embeddings.
- **1:1 Matching**: Upload an image and compare it with the last accessed `_live` image.
- **Retrieve Candidate Images**: Fetch the registered and live image of a candidate using their image_id.
- **Cosine Similarity**: Computes similarity between embeddings to determine identity confidence.

