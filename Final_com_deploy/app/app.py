import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from neuralnet import NeuralNet 
from preprocessingPipeline import PreprocessingPipeline 

colunas_treinadas = [
    'Tempo_conta',
    'Plano_internacional',
    'Plano_correio',
    'Total_minutos_diurnos',
    'Total_chamadas_diurnas',
    'Total_minutos_vespertinos',
    'Total_chamadas_vespertinas',
    'Total_minutos_noturnos',
    'Total_chamadas_noturnas',
    'Total_minutos_internacionais',
    'Total_chamadas_internacionais',
    'Chamadas_atendimento',
    'Media_cancelamento_por_estado',
    'Media_cancelamento_por_codigo_area',
    'Total_chamadas_noturnas_por_minuto'
]

colunas_para_normalizar = [
    'Tempo_conta', 'Total_minutos_diurnos', 'Total_chamadas_diurnas',
    'Total_minutos_vespertinos', 'Total_chamadas_vespertinas',
    'Total_minutos_noturnos', 'Total_chamadas_noturnas',
    'Total_minutos_internacionais'
]


@st.cache_resource
def carregar_modelo():
    modelo = NeuralNet()
    modelo.load_state_dict(torch.load("../models/modelo_redes_neurais.pt", map_location=torch.device('cpu')))
    modelo.eval()
    return modelo


st.title("Classificador com Rede Neural")
st.write("Preencha os dados para obter a classificação.")
st.success(f"Acurácia de: 91%; F1-score: 59%")

# Interface com entradas (adapte conforme suas colunas reais)
entrada = {
    "Plano_internacional": st.selectbox("O status do plano internacional do cliente", ["Sim", "Não"]),
    "Plano_correio": st.selectbox("O status do plano de correio de voz do cliente", ["Sim", "Não"]),
    "Chamadas_atendimento": st.number_input("Número total de chamadas feitas ao serviço de atendimento ao cliente", value=5),
    "Numero_mensagens_voz": st.number_input("Número de mensagens de correio de voz enviadas pelo cliente", value=90),
    "Tempo_conta": st.number_input('Número de dias usando os serviços', value=90),
    "Total_minutos_diurnos": st.number_input("Total de minutos de chamadas feitos por um cliente durante o dia", value=100),
    "Total_chamadas_diurnas": st.number_input("Número total de chamadas feitas por um cliente durante o dia", value=5),
    "Total_minutos_vespertinos": st.number_input("Total de minutos de chamadas feitos por um cliente durante a tarde", value=89),
    "Total_chamadas_vespertinas": st.number_input("Número total de chamadas feitas por um cliente durante a tarde", value=12),
    "Total_minutos_noturnos": st.number_input("Total de minutos de chamadas feitos por um cliente durante a noite", value=120),
    "Total_chamadas_noturnas": st.number_input("Número total de chamadas feitas por um cliente durante a noite", value=2),
    "Total_minutos_internacionais": st.number_input(" Total de minutos de chamadas internacionais feitas por um cliente", value=0),
    "Total_chamadas_internacionais": st.number_input("Número total de chamadas internacionais feitas por um cliente", value=0),
    "Total_cobrancas_diurnas": st.number_input("Valor total cobrado a um cliente durante o dia", value=120), 
    "Total_cobrancas_noturnas": st.number_input("Valor total cobrado a um cliente durante a noite", value=80),
    "Total_cobrancas_vespertinas": st.number_input("Valor total cobrado a um cliente durante a tarde", value=96),
    "Total_cobrancas_internacionais": st.number_input("Valor total cobrado por chamadas internacionais feitas por um cliente", value=0),
}

if st.button("Classificar"):
    try:
        
        df_entrada = pd.DataFrame([entrada])
        pipeline = PreprocessingPipeline()
        dados_transformados = pipeline.transform(df_entrada)
        dados_transformados = dados_transformados.reindex(columns=colunas_treinadas, fill_value=0)
        
        #scaler = joblib.load('../models/scaler.pkl')
        #df_normalizado = pd.DataFrame(scaler.transform(dados_transformados), columns=colunas_para_normalizar)

        modelo = carregar_modelo()
        tensor = torch.tensor(dados_transformados.values, dtype=torch.float32)
        with torch.no_grad():
            output = modelo(tensor).numpy()[0][0]

        classe = "Cancelará" if output > 0.5 else "Não Cancelará"
        
        st.success(f"Resultado: **{classe}** (score: {output:.2f})")
    except Exception as e:
        st.error(f"Erro na predição: {e}")
