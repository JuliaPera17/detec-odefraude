# Importar bibliotecas necessárias
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Carregar o arquivo CSV no Google Colab
df = pd.read_csv('/content/drive/MyDrive/Transações_Bancarias.csv')

print("Dataset Original:")
print(df.head())

#Retirar colunas que não serão usadas para o treino
df.drop(columns=['id_produto', 'desconto', 'FLG_DESCONTO', 'desconto.1', 'id_venda'], inplace=True)

print("Dataset Original:")
print(df.head())

# Converter a coluna de datetime para DateTime e criar uma coluna para identificar transações de madrugada
df['datetime_compra'] = pd.to_datetime(df['datetime_compra'])
df['is_madrugada'] = df['datetime_compra'].dt.hour.between(0, 5)

# Converter a coluna de datetime para DateTime e criar uma coluna para identificar transações de madrugada
df['datetime_compra'] = pd.to_datetime(df['datetime_compra'])
df['is_madrugada'] = df['datetime_compra'].dt.hour.between(0, 5)

# Definir um limite para "baixo valor" (exemplo: R$ 150)
baixo_valor_limite = 150
df['is_baixo_valor'] = df['valor'] < baixo_valor_limite

# Classificar como fraude se for "baixo valor" e "madrugada"
df['fraude'] = df['is_madrugada'] & df['is_baixo_valor']

# 3. Aplicar Label Encoding nas colunas especificadas
le_metodo_pagamento = LabelEncoder()
df['metodo_pagamento_encoded'] = le_metodo_pagamento.fit_transform(df['metodo_pagamento'].astype(str))

le_faixa_dia = LabelEncoder()
df['FAIXA_DO_DIA_encoded'] = le_faixa_dia.fit_transform(df['FAIXA_DO_DIA'].astype(str))

le_dia_semana = LabelEncoder()
df['DIA_DA_SEMANA_encoded'] = le_dia_semana.fit_transform(df['DIA_DA_SEMANA'].astype(str))

le_fbc_metodo_pagamento = LabelEncoder()
df['FBC_METODO_PAGAMENTO_encoded'] = le_fbc_metodo_pagamento.fit_transform(df['FBC_METODO_PAGAMENTO'].astype(str))

# Selecionar as colunas de entrada (features) e o alvo (fraude)
X = df[['valor', 'is_madrugada', 'is_baixo_valor', 'metodo_pagamento_encoded',
        'FAIXA_DO_DIA_encoded', 'DIA_DA_SEMANA_encoded', 'FBC_METODO_PAGAMENTO_encoded']]
y = df['fraude']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Treinar o modelo de classificação - Exemplo com RandomForest
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# 5. Fazer previsões e avaliar o modelo
y_pred = modelo.predict(X_test)
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("Acurácia:", accuracy_score(y_test, y_pred))

# 6. Reverter as colunas codificadas para valores originais após o modelo
df['metodo_pagamento'] = le_metodo_pagamento.inverse_transform(df['metodo_pagamento_encoded'])
df['FAIXA_DO_DIA'] = le_faixa_dia.inverse_transform(df['FAIXA_DO_DIA_encoded'])
df['DIA_DA_SEMANA'] = le_dia_semana.inverse_transform(df['DIA_DA_SEMANA_encoded'])
df['FBC_METODO_PAGAMENTO'] = le_fbc_metodo_pagamento.inverse_transform(df['FBC_METODO_PAGAMENTO_encoded'])

# Exibir dataset com colunas revertidas para valores originais
print("\nDataset com Colunas Revertidas para Valores Originais:")
print(df[['metodo_pagamento', 'FAIXA_DO_DIA', 'DIA_DA_SEMANA', 'FBC_METODO_PAGAMENTO', 'fraude']].head())

import matplotlib.pyplot as plt

# Fazer previsões no conjunto completo e adicionar ao dataframe
df['previsao_fraude'] = modelo.predict(X)

# Filtrar transações detectadas como fraude
df_fraudes = df[df['previsao_fraude'] == 1]

# Exibir dataset final com a coluna de previsão de fraudes
print("\nDataset Final com Previsão de Fraudes:")
print(df[['valor', 'datetime_compra', 'fraude', 'previsao_fraude']].head())

# Gráfico de Horas e Valores das Fraudes
plt.figure(figsize=(10, 6))
plt.scatter(df_fraudes['datetime_compra'].dt.hour, df_fraudes['valor'], color='red', label='Fraude')
plt.xlabel('Hora do Dia')
plt.ylabel('Valor da Transação')
plt.title('Horas e Valores das Transações Fraudulentas')
plt.legend()
plt.grid(True)
plt.show()

#Grafico em colunas
import matplotlib.pyplot as plt

# Agrupar fraudes por hora do dia
fraudes_por_hora = df_fraudes['datetime_compra'].dt.hour.value_counts().sort_index()

# Criar o gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(fraudes_por_hora.index, fraudes_por_hora.values, color='red', alpha=0.7)
plt.xlabel('Hora do Dia')
plt.ylabel('Quantidade de Fraudes')
plt.title('Quantidade de Fraudes por Hora do Dia')
plt.xticks(range(0, 24))  # Configurar o eixo x para mostrar todas as horas de 0 a 23
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Exibir o gráfico
plt.show()

