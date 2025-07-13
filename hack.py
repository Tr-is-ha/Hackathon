from flask import Flask, request, jsonify
import pytesseract
import cv2
import numpy as np
import requests
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

def pixels_to_cm(pixels, dpi=96):
    return round((pixels / dpi) * 2.54, 2)

def extract_poster_data(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Image decoding failed"}

    # Save downloaded image
    cv2.imwrite("downloaded_image.jpg", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

    min_x = min_y = float('inf')
    max_x = max_y = 0
    word_count = 0
    confidences = []

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = data['conf'][i]
        if text and conf != '-1':
            x = int(data['left'][i])
            y = int(data['top'][i])
            w = int(data['width'][i])
            h = int(data['height'][i])

            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

            word_count += 1
            confidences.append(int(conf))

    if word_count == 0:
        return {
            "containsPoster": False,
            "boundingBox": {},
            "dimensionsCm": {},
            "confidence": 0.0
        }

    bounding_box = {
        "x": min_x,
        "y": min_y,
        "width": max_x - min_x,
        "height": max_y - min_y
    }

    height, width = img.shape[:2]
    dimensions_cm = {
        "width": pixels_to_cm(width),
        "height": pixels_to_cm(height)
    }

    avg_conf = round(sum(confidences) / len(confidences), 2)

    return {
        "containsPoster": True,
        "boundingBox": bounding_box,
        "dimensionsCm": dimensions_cm,
        "confidence": round(avg_conf / 100, 2)
    }

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is working"})

@app.route('/validate-image', methods=['POST'])
def validate_image():
    try:
        image_bytes = None

        if request.is_json:
            data = request.get_json()
            image_url = data.get('imageUrl')
            if not image_url:
                return jsonify({"error": "Missing imageUrl in JSON"}), 400

            headers = {'User-Agent': 'Mozilla/5.0'}

            



            response = requests.get(image_url, headers=headers, timeout=10)
            response.raise_for_status()
            image_bytes = BytesIO(response.content).read()

            # ✅ DEBUG PRINTS
            print("✅ Image URL:", image_url)
            print("✅ Image size (bytes):", len(image_bytes))

            with open("downloaded_image.jpg", "wb") as f:
                f.write(image_bytes)

            print("✅ Image saved as downloaded_image.jpg")

        elif 'image' in request.files:
            image_file = request.files['image']
            image_bytes = image_file.read()

        else:
            return jsonify({"error": "No valid input provided. Use imageUrl or upload file."}), 400

        result = extract_poster_data(image_bytes)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
