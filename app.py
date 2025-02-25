import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify

app = Flask(__name__)

def prediction_call(crop_name, growth_phase, temp, humidity, precipitation, moisture, api_temp, api_humidity, api_precipitation):
    try:
        # Load datasets
        df_concat = pd.read_csv("datasets/deeplearning.csv")
        crop_df = pd.read_csv("datasets/crop_data.csv")
    except FileNotFoundError as e:
        return f"Error: {e}. Please ensure the dataset files exist."
    except Exception as e:
        return f"Unexpected error while loading data: {e}"

    try:
        # Filter crop data
        filtered_df = crop_df[
            (crop_df["Crop"].str.strip().str.lower() == crop_name.strip().lower()) & 
            (crop_df["Growth Phase"].str.strip().str.lower() == growth_phase.strip().lower())
        ]
        if filtered_df.empty:
            return "No matching data found. Please check the crop name and growth phase."
        water_requirement = filtered_df["Water Requirement (mm/day)"].values[0]
    except Exception as e:
        return f"Error processing crop data: {e}"

    try:
        # Feature Engineering
        df = df_concat.copy()
        df['MoistureContent_mm_Lag1'] = df['MoistureContent_mm'].shift(1)
        df['MoistureContent_mm_Lag2'] = df['MoistureContent_mm'].shift(2)
        df['MoistureContent_mm_RollingMean'] = df['MoistureContent_mm'].rolling(window=3).mean()
        df['MoistureContent_mm_RollingStd'] = df['MoistureContent_mm'].rolling(window=3).std()
        df['Temp_Humidity'] = df['Temperature_C'] * df['Humidity']
        df.dropna(inplace=True)
    except Exception as e:
        return f"Error in feature engineering: {e}"

    try:
        # Replaced list with a tuple to make it hashable
        features = (
            'Temperature_C', 'Humidity', 'Precipitation_mm', 'MoistureContent_mm_Lag1',
            'MoistureContent_mm_Lag2', 'MoistureContent_mm_RollingMean', 'MoistureContent_mm_RollingStd', 'Temp_Humidity'
        )  # Changed from list to tuple
        X = df[features]
        y = df['MoistureContent_mm']
        scaler = StandardScaler()
        scaler.fit(X)  # Fit the scaler on the feature data
        X_scaled = scaler.transform(X)
        X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
    except Exception as e:
        return f"Error in data preprocessing: {e}"

    try:
        # Load Pre-Trained Model
        model = load_model("trained_model.keras")
    except FileNotFoundError as e:
        return f"Error: {e}. Please ensure the trained model file exists."
    except Exception as e:
        return f"Unexpected error while loading model: {e}"

    # Moisture Prediction Function
    def predict_next_day_moisture(current_data):
        try:
            current_data_scaled = scaler.transform(current_data)
            current_data_scaled = np.reshape(current_data_scaled, (current_data_scaled.shape[0], 1, current_data_scaled.shape[1]))
            predicted_moisture = model.predict(current_data_scaled)
            return predicted_moisture[0][0]
        except Exception as e:
            return f"Prediction error: {e}"

    # Irrigation Recommendation Function
    def recommend_irrigation(total_predicted_moisture):
        try:
            if total_predicted_moisture < 1:
                return "Irrigate"
            elif total_predicted_moisture > 7:
                return "Do not irrigate, soil moisture sufficient"
            else:
                return "Monitor moisture levels"
        except Exception as e:
            return f"Error generating recommendation: {e}"

    try:
        # Day 0 Inputs (Using tuples instead of lists for hashable types)
        current_temperature = temp
        current_humidity = humidity
        current_precipitation = precipitation
        current_moisture = moisture
        total_predicted_moisture = 0
        
        # Ensure API inputs are tuples to avoid unhashable errors (Convert lists to tuples)
        api_temp = tuple(api_temp)  # Convert to tuple
        api_humidity = tuple(api_humidity)  # Convert to tuple
        api_precipitation = tuple(api_precipitation)  # Convert to tuple
        
        # Predict for the next 4 days
        for day in range(4):
            current_data = pd.DataFrame({
                'Temperature_C': [current_temperature],
                'Humidity': [current_humidity],
                'Precipitation_mm': [current_precipitation],
                'MoistureContent_mm_Lag1': [current_moisture],
                'MoistureContent_mm_Lag2': [current_moisture],
                'MoistureContent_mm_RollingMean': [current_moisture],
                'MoistureContent_mm_RollingStd': [current_moisture],
                'Temp_Humidity': [current_temperature * current_humidity]
            })
            current_data.fillna(current_moisture, inplace=True)
            predicted_moisture = predict_next_day_moisture(current_data)
            total_predicted_moisture += predicted_moisture  # Update for the next day's prediction
            current_moisture = predicted_moisture
            if day < len(api_temp):
                current_temperature = api_temp[day]
                current_humidity = api_humidity[day]
                current_precipitation = api_precipitation[day]
    except Exception as e:
        return f"Error during prediction loop: {e}"

    try:
        recommendation = recommend_irrigation(total_predicted_moisture)
        return f"Total Predicted Moisture over 4 days: {total_predicted_moisture:.2f}, Recommendation: {recommendation}"
    except Exception as e:
        return f"Error in final result processing: {e}"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = prediction_call(
        data['crop_name'],
        data['growth_phase'],
        data['temp'],
        data['humidity'],
        data['precipitation'],
        data['moisture'],
        data['api_temp'],
        data['api_humidity'],
        data['api_precipitation']
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
