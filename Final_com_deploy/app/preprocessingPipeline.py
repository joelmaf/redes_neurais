import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class PreprocessingPipeline:
    def __init__(self):
        self.label_encoders = {
            'Plano_internacional': LabelEncoder(),
            'Plano_correio': LabelEncoder(),
            'Chamadas_atendimento': LabelEncoder(),
            'Numero_mensagens_voz': LabelEncoder()
        }

    def binning(self, df,  bins, labels, column):
        if bins is not None and labels is not None:
            df[column] = pd.cut(df[column], bins=bins, labels=labels, right=False)
        return df
        
    def runbinning(self, df):
        bins = [0, 3, 5, np.inf]
        labels = ['Baixo', 'Médio', 'Alto']
        df = self.binning(df, bins, labels, 'Chamadas_atendimento')
        bins = [0, 21, np.inf]
        labels = ['Baixo', 'Alto']
        df = self.binning(df, bins, labels, 'Numero_mensagens_voz')
        return df

    def log_transformation(self, df):
        df['Total_chamadas_internacionais'] = np.log1p(df['Total_chamadas_internacionais'])
        return df

    def label_encoder(self, df):
        le = LabelEncoder()
        df['Plano_internacional'] = le.fit_transform(df['Plano_internacional'])
        df['Plano_correio'] = le.fit_transform(df['Plano_correio'])
        df['Chamadas_atendimento'] = le.fit_transform(df['Chamadas_atendimento'])
        df['Numero_mensagens_voz'] = le.fit_transform(df['Numero_mensagens_voz'])
        return df

    def feature_engineering(self, df):
        df['Total_chamadas_diurnas_por_minuto'] = df['Total_cobrancas_diurnas'] / df['Total_minutos_diurnos']
        df['Total_chamadas_vespertinas_por_minuto'] = df['Total_cobrancas_vespertinas'] / df['Total_minutos_vespertinos']
        df['Total_chamadas_noturnas_por_minuto'] = df['Total_cobrancas_noturnas'] / df['Total_minutos_noturnos']
        df['Total_chamadas_internacionais_por_minuto'] = df['Total_cobrancas_internacionais'] / df['Total_minutos_internacionais']
        return df

    def transform(self, df):
        df = self.runbinning(df)
        df = self.log_transformation(df)
        df = self.feature_engineering(df)
        df = self.label_encoder(df)
        return df
