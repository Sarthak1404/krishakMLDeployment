import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Define request body using Pydantic
class PredictionRequest(BaseModel):
    crop_name: str
    growth_phase: str
    temp: float
    humidity: float
    precipitation: float
    moisture: float
    api_temp: List[float]
    api_humidity: List[float]
    api_precipitation: List[float]

def prediction_call(crop_name, growth_phase, temp, humidity, precipitation, moisture, api_temp, api_humidity, api_precipitation):
    try:
        # Load datasets
        df_concat = pd.read_csv("datasets/deeplearning.csv")
        crop_df = pd.read_csv("datasets/crop_data.csv")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Dataset files not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error loading data: {e}")

    try:
        # Filter crop data
        filtered_df = crop_df[
            (crop_df["Crop"].str.strip().str.lower() == crop_name.strip().lower()) & 
            (crop_df["Growth Phase"].str.strip().str.lower() == growth_phase.strip().lower())
        ]
        if filtered_df.empty:
            raise HTTPException(status_code=400, detail="No matching data found for the given crop and growth phase.")
        water_requirement = filtered_df["Water Requirement (mm/day)"].values[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing crop data: {e}")

    try:
        # Feature Engineering
        df = df_concat.copy()
        df['MoistureContent_mm_Lag1'] = df['MoistureContent_mm'].shift(1)
        df['MoistureContent_mm_Lag2'] = df['MoistureContent_mm'].shift(2)
        df['MoistureContent_mm_RollingMean'] = df['MoistureContent_mm'].rolling(window=2).mean()
        df['MoistureContent_mm_RollingStd'] = df['MoistureContent_mm'].rolling(window=2).std()
        df['Temp_Humidity'] = df['Temperature_C'] * df['Humidity']
        df.dropna(inplace=True)

        features = [
            'Temperature_C', 'Humidity', 'Precipitation_mm', 'MoistureContent_mm_Lag1',
            'MoistureContent_mm_Lag2', 'MoistureContent_mm_RollingMean', 'MoistureContent_mm_RollingStd', 'Temp_Humidity'
        ]
        X = df[features]
        scaler = StandardScaler()
        scaler.fit(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in data preprocessing: {e}")

    try:
        # Load Pre-Trained Model
        model = load_model("trained_model.keras")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error loading model: {e}")

    def predict_next_day_moisture(current_data):
        try:
            current_data_scaled = scaler.transform(current_data)
            current_data_scaled = np.reshape(current_data_scaled, (current_data_scaled.shape[0], 1, current_data_scaled.shape[1]))
            predicted_moisture = model.predict(current_data_scaled)
            return predicted_moisture[0][0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    def recommend_irrigation(total_predicted_moisture):
        try:
            if (total_predicted_moisture//3) < 1:
                return "Irrigate"
            elif (total_predicted_moisture//3) > 7:
                return "Do not irrigate, soil moisture sufficient"
            else:
                return "Monitor moisture levels"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating recommendation: {e}")

    try:
        current_temperature = temp
        current_humidity = humidity
        current_precipitation = precipitation
        current_moisture = moisture
        total_predicted_moisture = 0

        for day in range(3):
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
            total_predicted_moisture += predicted_moisture
            current_moisture = predicted_moisture

            if day < len(api_temp):
                current_temperature = api_temp[day]
                current_humidity = api_humidity[day]
                current_precipitation = api_precipitation[day]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction loop: {e}")

    try:
        recommendation = recommend_irrigation(total_predicted_moisture)
        return {
            "total_predicted_moisture": round(total_predicted_moisture, 2),
            "recommendation": recommendation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in final result processing: {e}")

@app.post("/predict")
async def predict(request: PredictionRequest):
    result = prediction_call(
        request.crop_name,
        request.growth_phase,
        request.temp,
        request.humidity,
        request.precipitation,
        request.moisture,
        request.api_temp,
        request.api_humidity,
        request.api_precipitation
    )
    return result

# Run the app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)