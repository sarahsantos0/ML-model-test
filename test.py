# Esse é um projeto teste de uma LLM usando regressão linear para prever o resultado da prova de um estudante 
# baseado em suas horas de estudo.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Criação de um dataset de exemplo com horas de estudo e notas da prova
data = {
    'Horas_de_Estudo': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Nota_da_Prova': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}

# 2. Conversão do dicionário em um DataFrame do pandas
df = pd.DataFrame(data)

# 3. Separação das variáveis independentes (X) e dependentes (y)
X = df[['Horas_de_Estudo']] # Entrada: horas de estudo (formato 2D)
y = df['Nota_da_Prova']     # Saída: nota da prova

# 4. Divisão dos dados em conjuntos de treino e teste (80% treino, 20% teste)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Criação e treinamento do modelo de regressão linear
model = LinearRegression()
model.fit(x_train, y_train)

# 6. Realização de previsões usando o conjunto de teste
y_pred = model.predict(x_test)
print(f'Previsões: {y_pred}')

# 7. Avaliação do modelo usando o erro quadrático médio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Erro Quadrático Médio: {mse}')

# 8. Visualização dos resultados
# Converte X para array 1D para facilitar a plotagem
X_1d = X['Horas_de_Estudo'].values

# Gera as previsões para todos os dados para desenhar a linha de regressão
y_pred_full = model.predict(X).flatten()

# Plota os pontos reais (azul) e a linha de regressão (vermelha)
plt.scatter(X_1d, y, color='blue', label='Notas atuais')
plt.plot(X_1d, y_pred_full, color='red', label='Linha de Regressão')
plt.xlabel('Horas de Estudo')
plt.ylabel('Nota da Prova')
plt.title('Regressão Linear: Horas de Estudo vs Nota da Prova')
plt.legend()
plt.show()