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

![Matriz de Confusão RandomForest](https://github.com/seu-usuario/seu-repositorio/blob/main/images/matriz_confusao_rf.png.png?raw=true) <!-- Certifique-se de que o caminho para a imagem esteja correto no seu repositório GitHub -->


## 🔮 Classificando Novos Clientes

A seção final do notebook demonstra como usar o pipeline de pré-processamento e o modelo treinado para classificar um novo cliente e prever sua probabilidade de churn.

```python
# Exemplo de predição para um novo cliente
# [Adicione um GIF ou um screenshot aqui mostrando a predição de um novo cliente, por exemplo, o output da célula s5bxoM-k86Zy]
```

O modelo prevê se o cliente irá cancelar o serviço e com qual probabilidade, auxiliando na tomada de decisões estratégicas.

## 🤝 Contribuições

Contribuições são bem-vindas! Se você tiver sugestões, melhorias ou encontrar algum problema, sinta-se à vontade para abrir uma issue ou enviar um pull request.
