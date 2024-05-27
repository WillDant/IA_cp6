from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Inicializar a aplicação FastAPI
app = FastAPI()

# Carregar o modelo treinado e o encoder
model = joblib.load('telecom_model.pkl')
encoder = joblib.load('encoder.pkl')

# Definir a estrutura dos dados do usuário
class UserInput(BaseModel):
    CustomerID: str
    Gender: str
    Age: int
    Married: str
    Number_of_Dependents: int
    City: str
    Zip_Code: int
    Latitude: float
    Longitude: float
    Number_of_Referrals: int
    Tenure_in_Months: int
    Offer: str
    Phone_Service: str
    Avg_Monthly_Long_Distance_Charges: float
    Multiple_Lines: str
    Internet_Service: str
    Internet_Type: str
    Avg_Monthly_GB_Download: float
    Online_Security: str
    Online_Backup: str
    Device_Protection_Plan: str
    Premium_Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Streaming_Music: str
    Unlimited_Data: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str
    Monthly_Charge: float
    Total_Charges: float
    Total_Refunds: float
    Total_Extra_Data_Charges: float
    Total_Long_Distance_Charges: float
    Total_Revenue: float

# Endpoint de previsão
@app.post("/predict_churn/")
def predict_churn(user_input: UserInput):
    # Converter a entrada para DataFrame
    user_data = pd.DataFrame([user_input.dict()])
    
    # Codificar os dados do usuário
    user_data_encoded = encoder.transform(user_data)
    
    # Calcular a probabilidade de churn
    churn_prob = model.predict_proba(user_data_encoded)[:, 1][0]
    
    # Retornar a probabilidade de churn
    return {"churn_probability": churn_prob}

# Rodar o servidor com Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
