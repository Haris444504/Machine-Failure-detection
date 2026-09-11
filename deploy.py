import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load the trained model pipeline
model_pipeline = joblib.load("Maintenance Preditive Intelligence.pkl")


class InputData(BaseModel):
    UDI: int
    Product_ID: str
    Type: str
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float
    HDF_Risk: int
    Failure_Risk: int


@app.post("/predict")
def predict(data: InputData):
    try:
        # 1. Map API JSON payload to DataFrame column names matching training data
        raw_data = {
            "UDI": [data.UDI],
            "Product ID": [data.Product_ID],
            "Type": [data.Type],
            "Air temperature [K]": [data.Air_temperature_K],
            "Process temperature [K]": [data.Process_temperature_K],
            "Rotational speed [rpm]": [data.Rotational_speed_rpm],
            "Torque [Nm]": [data.Torque_Nm],
            "Tool wear [min]": [data.Tool_wear_min],
            "HDF_Risk": [data.HDF_Risk],
            "Failure_Risk": [data.Failure_Risk],
        }

        df = pd.DataFrame(raw_data)

        # 2. Re-create EXACT feature engineering step from Notebook Cell [11]
        df["temperature_difference"] = (
            df["Process temperature [K]"] - df["Air temperature [K]"]
        )

        df["Power"] = (
            df["Rotational speed [rpm]"]
            * df["Torque [Nm]"]
            * 2
            * (3.12 / 60)
        )

        df["Energy Loss"] = (
            df["Rotational speed [rpm]"] * df["Tool wear [min]"]
        )

        df["Overstrain_Index"] = df["Torque [Nm]"] * df["Tool wear [min]"]

        df["Failure_Risk"] = 0
        df.loc[(df["Power"] < 3500) | (df["Power"] > 9000), "Failure_Risk"] = 1

        df["HDF_Risk"] = (
            (df["temperature_difference"] < 8.6)
            & (df["Rotational speed [rpm]"] < 1380)
        ).astype(int)

        # 3. Make Prediction
        prediction = model_pipeline.predict(df)[0]
        probabilities = model_pipeline.predict_proba(df)[0].tolist()

        return {
            "prediction": int(prediction),
            "status": "Failure" if prediction == 1 else "No Failure",
            "probabilities": {
                "no_failure": probabilities[0],
                "failure": probabilities[1],
            },
        }

    except Exception as e:
        # Print exact error to terminal console for debugging
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))