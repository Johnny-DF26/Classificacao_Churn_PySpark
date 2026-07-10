# 📊 Previsão de Churn de Clientes com PySpark

Este projeto demonstra a construção de um modelo de Machine Learning capaz de classificar clientes entre possíveis cancelamentos (churn) ou não, utilizando o framework PySpark para processamento e modelagem de dados em grande escala. O objetivo é identificar clientes com maior probabilidade de cancelar, permitindo que as equipes de marketing e relacionamento tomem ações proativas.

## ✨ Funcionalidades

- **Carregamento de Dados:** Leitura e inspeção inicial de um dataset de clientes.
- **Pré-processamento:** Tratamento de valores binários e categóricos com `StringIndexer`, `OneHotEncoder` e `VectorAssembler` do PySpark MLlib.
- **Pipeline de Dados:** Criação de um pipeline para automatizar as etapas de pré-processamento.
- **Divisão Treino/Teste:** Separação do dataset para treinamento e avaliação do modelo.
- **Modelagem:** Implementação e treinamento de modelos de `LogisticRegression` e `RandomForestClassifier`.
- **Avaliação:** Análise de desempenho dos modelos utilizando `classification_report` e matrizes de confusão.
- **Padronização de Features:** Avaliação do impacto do `StandardScaler` no desempenho da `LogisticRegression`.
- **Otimização de Hiperparâmetros:** Uso de `CrossValidator` e `ParamGridBuilder` para tunar o `RandomForestClassifier`.
- **Persistência de Modelos:** Salvamento e carregamento do pipeline de pré-processamento e do melhor modelo treinado.
- **Predição em Novos Dados:** Demonstração de como classificar um novo cliente com o modelo final.

## 🛠️ Tecnologias Utilizadas

- **PySpark:** Para processamento distribuído e Machine Learning.
- **Apache Spark MLlib:** Componente de Machine Learning do Spark.
- **Pandas:** Para manipulação de DataFrames em algumas etapas de visualização.
- **Scikit-learn:** Para métricas de avaliação (`classification_report`, `ConfusionMatrixDisplay`).
- **Matplotlib:** Para visualização de dados (matrizes de confusão).
- **Google Colab:** Ambiente de desenvolvimento.

## 🚀 Como Rodar o Projeto

### Instalação e Configuração

1.  **Ambiente:** O projeto foi desenvolvido e testado no Google Colab. Recomenda-se utilizá-lo para uma experiência mais fluida, pois já vem com muitos requisitos pré-instalados.
2.  **PySpark:** A primeira célula do notebook instala o PySpark:
    ```bash
    !pip install pyspark
    ```
3.  **Download dos Dados:** Certifique-se de que o arquivo `dados_clientes.csv` esteja presente no diretório `/content/` do seu ambiente Colab, ou ajuste o caminho de leitura do DataFrame.

### Execução

Basta abrir o notebook (`seu_projeto_churn.ipynb`) no Google Colab e executar as células sequencialmente. O notebook é dividido em seções para facilitar a compreensão:

1.  **Base de Dados:** Carregamento e exploração inicial.
2.  **Transformando os Dados:** Pré-processamento e feature engineering.
3.  **Machine Learning:** Treinamento, avaliação e otimização de modelos.
4.  **Modelo Final:** Salvamento do modelo e previsão de novos dados.

## 📈 Resultados e Conclusões

O `RandomForestClassifier` otimizado via `CrossValidator` demonstrou a melhor performance, com uma acurácia de **~81.83%** no conjunto de teste. A padronização dos dados com `StandardScaler` não resultou em ganhos significativos para a `LogisticRegression` neste dataset.

### Matriz de Confusão do Modelo Final (RandomForest)

![Matriz de Confusão RandomForest](https://github.com/Johnny-DF26/Classificacao_Churn_PySpark/blob/main/images/matriz_confusao_rf.png?raw=true)


## 🔮 Classificando Novos Clientes

Este guia descreve como utilizar o pipeline de pré-processamento e o modelo treinado para classificar o churn de novos clientes.

### 1. Preparação

Certifique-se de ter os seguintes arquivos salvos:
- `pipeline_preprocess_churn` (Pipeline de pré-processamento)
- `classificador_rf_final_churn` (Modelo de Classificação RandomForest)
- `schema_churn.json` (Esquema dos dados originais)

### 2. Carregar Pipeline e Modelo

Carregue o pipeline de pré-processamento e o modelo de classificação salvos:

```python
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
import json
from pyspark.sql.types import StructType

pipeline_preprocess = PipelineModel.load("pipeline_preprocess_churn")
rf_model = RandomForestClassificationModel.load("classificador_rf_final_churn")

with open("/content/schema_churn.json", "r") as f:
    schema = StructType.fromJson(json.load(f))
```

### 3. Preparar Novo Cliente para Predição

Crie um DataFrame Spark com os dados do novo cliente, utilizando o esquema carregado, e aplique o pipeline de pré-processamento:

```python
novo_cliente_data = [
    (
        0, "Sim", "Sim", 36, "Nao", "Nao",
        "FibraOptica", "Nao", "Sim", "Nao",
        "Sim", "Nao", "Sim",
        "Mensalmente", "Nao", "Boleto", 45.0
    )
]
novo_cliente_df = spark.createDataFrame(novo_cliente_data, schema=schema)

cliente_processado = pipeline_preprocess.transform(novo_cliente_df)
cliente_processado.show()
```

### 4. Realizar Predição

Utilize o modelo treinado para prever o churn do cliente e interprete o resultado:

```python
predicao = rf_model.transform(cliente_processado)
valor_predicao = predicao.select("prediction").first()[0]
probabilidade = predicao.select("probability").first()[0]

print(f'Classificação:')
if valor_predicao == 0.0:
    print(f"Cliente NÃO irá cancelar o serviço com {probabilidade[0]*100:.2f}%")
else:
    print(f"Cliente irá cancelar o serviço {probabilidade[1]*100:.2f}%")
```

Este processo permite a classificação rápida de novos clientes para identificar o risco de churn.

O modelo prevê se o cliente irá cancelar o serviço e com qual probabilidade, auxiliando na tomada de decisões estratégicas.

## 🤝 Contribuições

Contribuições são bem-vindas! Se você tiver sugestões, melhorias ou encontrar algum problema, sinta-se à vontade para abrir uma issue ou enviar um pull request.
