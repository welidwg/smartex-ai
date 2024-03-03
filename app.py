import easyocr
import numpy
from flask import Flask, request, jsonify
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

reader=easyocr.Reader(["fr"],gpu=False)
@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        image = cv2.imdecode(numpy.frombuffer(request.files['image'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
        text_result=reader.readtext(image)
        text=[result[1] for result in text_result]
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
