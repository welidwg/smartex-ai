import easyocr
import numpy
from flask import Flask, request, jsonify
import cv2
from flask_cors import CORS
from datetime import datetime, timedelta
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from prophet import Prophet
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import time
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
laravelUrl = "http://127.0.0.1:8000/api"

reader = easyocr.Reader(["fr"], gpu=True)
delay_time = 1
@app.route('/prediction', methods=['POST'])
def prediction():
    try:
        body = request.json  # recupertion id machine
        id_machine = body['id_machine']
        history=body['history']
        if len(history) != 0:
            dataset = []
            for entry in history:
                data = {
                    'id_machine': entry['id_machine'],
                    'date_heure': entry['date_heure']
                }
                dataset.append(data)

            df_n = pd.DataFrame(dataset)
            df_n['date_heure'] = pd.to_datetime(df_n['date_heure'])
            df_n.sort_values(by='date_heure', inplace=True, ascending=True)
            df_n['diff_temps'] = df_n.groupby('id_machine')['date_heure'].diff().dt.total_seconds()
            df_n['diff_temps'] = df_n.groupby('id_machine')['diff_temps'].transform(lambda x: x.fillna(x.median()))

            model = LinearRegression()
            X = pd.DataFrame(df_n['id_machine'], columns=['id_machine'])
            y = df_n['diff_temps']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            nouvelles_donnees = pd.DataFrame({'id_machine': [id_machine]})
            prochaine_diff_temps = model.predict(nouvelles_donnees)[0]
            derniere_date_heure = df_n['date_heure'].max()
            delai_temps = timedelta(seconds=prochaine_diff_temps)
            avg = delai_temps.days+1
            print("delai_temps", avg)
            prochaine_date_heure_predite = derniere_date_heure + delai_temps
            date_string = prochaine_date_heure_predite.strftime("%a, %d %b %Y %H:%M:%S GMT")
            parsed_date = datetime.strptime(date_string, "%a, %d %b %Y %H:%M:%S %Z")
            formatted_date = parsed_date.strftime("%Y-%m-%d %H:%M:%S")

            mae = mean_absolute_error(y_test, y_pred)
            mae_percentage = (mae / y_test.mean()) * 100
            print(f"MAE : {mae_percentage:.4f}")

            return jsonify({"id_machine": id_machine, "next": formatted_date,"avg": avg})
        else:
            return jsonify({"message" : "no data available","next":""})

    except Exception as e:
        return jsonify({'message': str(e), 'type': "error"}), 500





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
        text_result = reader.readtext(cropped_image)
        text = [result[1] for result in text_result]
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
