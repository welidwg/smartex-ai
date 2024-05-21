import easyocr
import numpy
from flask import Flask, request, jsonify
import cv2
from flask_cors import CORS
from joblib import load
from datetime import datetime, timedelta
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from prophet import Prophet
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import time
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
laravelUrl = "http://127.0.0.1:8000/api"

reader = easyocr.Reader(["fr"], gpu=True)
delay_time = 1

@app.route('/')
def hello_world():

    return 'Hello World!'
@app.route('/machines',methods=['POST'])
def machines():
    try:
        file_path = 'machines.csv'
        try:
            df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, delimiter=';', encoding='latin1')

        df['parc'] = df['parc'].astype(int)
        api_url = f"{laravelUrl}/machine/flask"
        print(df.head())
        for index, row in df.iterrows():
            data = {
                'code': row['code_machine'],
                'id_etat': 1,
                'id_chaine': 2,
                'reference': row['reference'],
                "parc": row['parc']
            }
            response = requests.post(api_url, json=data)
            if response.status_code == 201:
                print(f"Record {index} created successfully.")
                print(response.text)
                #print(response.json())
            elif response.status_code == 429:
                print(f"Rate limit hit at record {index}. Status code: {response.status_code}. Retrying after delay.")
                time.sleep(delay_time)  # Wait before retrying
                continue  # Skip to the next iteration
            else:
                print(
                    f"Failed to create record {index}. Status code: {response.status_code}, Response: {response.text}")
            time.sleep(delay_time)
        return "Record created successfully"
    except Exception as e:
        print(e)
        return 'Something went wrong while processing'


@app.route('/op',methods=['POST'])
def operations():
    try:
        file_path = 'operations.csv'
        try:
            df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, delimiter=';', encoding='latin1')
        df['temps'] = df['temps'].str.replace(',', '.')
        df['temps'] = df['temps'].astype(float)
        api_url = f"{laravelUrl}/operation"
        print(df.head())
        for index, row in df.iterrows():
            data = {
                'libelle': row['libelle'],
                'reference': row['reference'],
                'id_gamme': row['id_gamme'],
                'temps': row['temps']
            }
            response = requests.post(api_url, json=data)
            if response.status_code == 201:
                print(f"Record {index} created successfully.")
                #print(response.json())
            elif response.status_code == 429:
                print(f"Rate limit hit at record {index}. Status code: {response.status_code}. Retrying after delay.")
                time.sleep(delay_time)  # Wait before retrying
                continue  # Skip to the next iteration
            else:
                print(
                    f"Failed to create record {index}. Status code: {response.status_code}, Response: {response.text}")
            time.sleep(delay_time)
        return "Record created successfully"
    except Exception as e:
        print(e)
        return 'Something went wrong while processing'



@app.route('/predicition/mll', methods=['POST'])
def prediciton_mll():
    try:
        body = request.json  # recupertion d'id machine
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

            # model = RandomForestRegressor()
            model = LinearRegression()
            X = pd.DataFrame(df_n['id_machine'], columns=['id_machine'])
            y = df_n['diff_temps']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mae_percentage = (mae / y_test.mean()) * 100
            print(f"MAE : {mae_percentage:.4f}")

            nouvelles_donnees = pd.DataFrame({'id_machine': [id_machine]})
            prochaine_diff_temps = model.predict(nouvelles_donnees)[0]
            derniere_date_heure = df_n['date_heure'].max()
            delai_temps = timedelta(seconds=prochaine_diff_temps)
            prochaine_date_heure_predite = derniere_date_heure + delai_temps
            date_string = prochaine_date_heure_predite.strftime("%a, %d %b %Y %H:%M:%S GMT")
            parsed_date = datetime.strptime(date_string, "%a, %d %b %Y %H:%M:%S %Z")
            formatted_date = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
            return jsonify({"id_machine": id_machine, "next": formatted_date})
        else:
            return jsonify({"message" : "no data available","next":""})

    except Exception as e:
        return jsonify({'message': str(e), 'type': "error"}), 500


@app.route('/prediciton/ml', methods=['POST'])
def prediciton_ml():
    try:
        body = request.json  # recupertion d'id machine
        id_machine = body['id_machine']
        response = requests.get(f"{laravelUrl}{id_machine}")
        dataset = []
        for entry in response.json():
            data = {
                'id_machine': entry['id_machine'],
                'date_heure': entry['date_heure']
            }
            dataset.append(data)
        df_n = pd.DataFrame(dataset)


        df_n['date_heure'] = pd.to_datetime(df_n['date_heure'])

        machine_data = df_n[df_n['id_machine'] == id_machine]
        machine_data.rename(columns={'date_heure': 'ds', 'id_machine': 'y'}, inplace=True)
        model = Prophet()
        model.fit(machine_data)
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)
        next_date = forecast['ds'].iloc[-1]
        return jsonify({"id_machine": id_machine, "next": next_date})

    except Exception as e:
        return jsonify({'message': str(e), 'type': "error"}), 500


@app.route('/predicition', methods=['POST'])
def prediciton_machine():
    try:
        body = request.json
        id_machine = body['id_machine']
        response = requests.get(f"{laravelUrl}{id_machine}")
        dataset = []
        for entry in response.json():
            data = {
                'id_machine': entry['id_machine'],
                'date_heure': entry['date_heure']
            }
            dataset.append(data)

        df_n = pd.DataFrame(dataset)
        print(df_n.head())

        df_n['date_heure'] = pd.to_datetime(df_n['date_heure'], dayfirst=False)
        df_n.sort_values(by='date_heure', inplace=True)

        df_n['time_diff'] = df_n.groupby('id_machine')['date_heure'].diff()
        df_n['time_diff'] = df_n.groupby('id_machine')['time_diff'].transform(lambda x: x.fillna(x.median()))
        avg_time_diff = df_n[df_n['id_machine'] == id_machine]['time_diff'].mean()

        last_timestamp = df_n[df_n['id_machine'] == id_machine]['date_heure'].max()

        next_timestamp = last_timestamp + avg_time_diff
        return jsonify({"id_machine": id_machine})

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
