import easyocr
import numpy
from flask import Flask, request, jsonify
import cv2
from flask_cors import CORS
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

reader=easyocr.Reader(["fr"],gpu=True)
@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        image = cv2.imdecode(numpy.frombuffer(request.files['image'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
        height, width, _ = image.shape
        red_rectangle_top = max(0, height // 2 - 80)  # Décalage vers le haut de 50 pixels
        red_rectangle_bottom = min(height, height // 2 + 80)  # Décalage vers le bas de 50 pixels
        red_rectangle_left = 0
        red_rectangle_right = width

        cropped_image = image[red_rectangle_top:red_rectangle_bottom, red_rectangle_left:red_rectangle_right]
        # plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        # plt.title('Image reçue dans Flask')
        # plt.axis('off')
        # plt.show()
        text_result=reader.readtext(cropped_image)
        text=[result[1] for result in text_result]
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
