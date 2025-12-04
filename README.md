# 📘 Class Score Prediction — Regressão Linear

Projeto simples e educacional que demonstra como treinar um modelo de Regressão Linear para prever a nota de uma prova com base nas horas de estudo.

## 🚀 Tecnologias utilizadas
- Python 3.8+
- numpy
- pandas
- scikit-learn
- matplotlib

## 🎯 Funcionalidades
- Criação de um dataset de exemplo (horas de estudo → nota).
- Treinamento de um modelo de regressão linear.
- Previsões sobre conjunto de teste.
- Cálculo do Erro Quadrático Médio (MSE).
- Visualização dos pontos reais e da linha de regressão.

## 📂 Estrutura do projeto
```markdown
.
├── test.py          # Script principal (dataset, treino, avaliação e gráficos)
└── README.md        # Documentação do projeto
```


## 🧠 Como ele funciona

1. Define um dicionário contendo duas colunas:

    - Horas_de_Estudo
    - Nota_da_Prova

2. Converte o dicionário em um ```pandas.DataFrame```
3. Separa:

    - X → feature (horas de estudo)
    - y → target (nota da prova)

4. Divide os dados em treino (80%) e teste (20%) com ```train_test_split```
5. Cria e treina um modelo LinearRegression.
6. Gera previsões no conjunto de teste e calcula o MSE com ```mean_squared_error```
7. Plota:

    - pontos reais (scatter)
    - linha de regressão aprendida pelo modelo

## Como executar (Windows)
```powershell
# opcional: criar e ativar ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate

# instalar dependências
pip install -r requirements.txt
# ou
pip install numpy pandas scikit-learn matplotlib

# executar
python test.py
```

Observação: os dados de exemplo são perfeitamente lineares; em casos reais utilize um dataset mais representativo e avalie múltiplas métricas e modelos.
