# Exemplo 2 - Wine Quality Fast API
from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Inicializar a aplicação FastAPI
app = FastAPI()

# Carregar o modelo treinado
model = joblib.load('wine_quality_model.pkl')

# Definir o modelo de dados para a entrada da API
class WineFeatures(BaseModel):
    winery: str
    wine: str
    year: int
    num_reviews: int
    country: str
    region: str
    price: float
    type: str
    body: int
    acidity: int

@app.post("/predict/")
async def predict(features: WineFeatures):
    # Converter a entrada para um DataFrame
    features_dict = features.dict()
    features_df = pd.DataFrame([features_dict])
    features_df = pd.get_dummies(features_df).reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Fazer a previsão
    prediction = model.predict(features_df)
    return {"prediction": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

