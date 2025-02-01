from flask import Flask, request, jsonify
from deepface import DeepFace
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import requests
import numpy as np

app = Flask(__name__)

@app.route("/extract", methods=["POST"])
def extract_embedding():
    try:
        data = request.json
        image_url = data["image"]

        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            return jsonify({"error": "Unable to download image"}), 400

        try:
            img = Image.open(BytesIO(response.content))
        except UnidentifiedImageError:
            return jsonify({"error": "Invalid image format"}), 400

        embedding = DeepFace.represent(img_path=np.array(img), model_name="Facenet")[0]["embedding"]

        return jsonify({"embedding": embedding})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Image download error: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Face detection or embedding generation failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(port=8000)
