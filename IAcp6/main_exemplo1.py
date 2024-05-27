# Exemplo 1 - Titanic Fast APi
from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Carregar modelo e scaler
with open('titanic_fare_model.pkl', 'rb') as f:
    model = pickle.load(f)

scaler = StandardScaler()
scaler.fit_transform(data[['Age']])

@app.post('/predict/')
def predict(data: dict):
    df = pd.DataFrame([data])
    df[['Age']] = scaler.transform(df[['Age']])  # Normalizar os dados
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
    
    # Adicionar colunas que podem estar faltando após get_dummies
    for col in X_train.columns:
        if col not in df.columns:
            df[col] = 0
            
    df = df[X_train.columns]  # Reordenar as colunas para garantir a mesma ordem
    
    prediction = model.predict(df)
    return {'prediction': prediction[0]}
