# 📊 SIOB - Data Science & Inteligência Estratégica

> **Módulo de Inteligência de Dados do Sistema Integrado de Ocorrências de Bombeiros (CBMPE).**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](COLE_SEU_LINK_DO_DEPLOY_AQUI)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/Status-Concluído-success)

---

## 📌 Sobre o Projeto

Este projeto é uma **extensão estratégica** do ecossistema SIOB. Enquanto o Painel Web foca no registro operacional em tempo real, este Dashboard foi desenvolvido para transformar dados históricos em **inteligência para tomada de decisão**.

Utilizando **Python** e bibliotecas de **Machine Learning**, o sistema analisa padrões de ocorrências, prevê tendências de consumo de recursos e gera alertas preventivos para o comando do Corpo de Bombeiros.

---

## 🧠 Funcionalidades de Inteligência Artificial

O diferencial deste projeto é a aplicação de algoritmos de Machine Learning para extrair insights ocultos nos dados:

### 1. Clusterização (K-Means) 🔵🟣
Agrupamento automático de ocorrências baseado em **Tempo de Resposta vs. Consumo de Água**.
- **Insight:** Identificou que ocorrências com tempo de resposta > 30 min (Clusters Críticos) tendem a triplicar o consumo de recursos hídricos.

### 2. Regressão Linear (OLS) 📈
Análise estatística da tendência entre tempo e prejuízo.
- **Insight:** Comprovação matemática de que o atraso no atendimento possui correlação positiva direta com a gravidade do incêndio.

### 3. Modelo Preditivo (Random Forest) 🌲
Algoritmo treinado para identificar os **Fatores Determinantes** no consumo de água.
- **Conclusão:** O modelo apontou que o **Tempo de Resposta** é o fator mais impactante (80%), superando a localização (Bairro) ou o Tipo de Incêndio.

---

## 🚒 Outros Módulos do Dashboard

* **🌎 Visão Geral:** Mapa de calor (distribuição espacial), análise de sazonalidade (linha do tempo) e Boxplot para auditoria de tempos de resposta.
* **🔥 Gestão de Incêndios:** Sistema de recomendação que emite **Alertas Automáticos** baseados em causas recorrentes (ex: Risco Crítico de Gás no Centro, Risco Sazonal de Vegetação).
* **🦈 Shark Monitor (Contexto Local):** Módulo específico para monitoramento de incidentes em praias, com análise de perfil de vítimas e alertas de tubarão.

---

## 🛠️ Tecnologias Utilizadas

| Categoria | Biblioteca / Ferramenta | Função |
| :--- | :--- | :--- |
| **Linguagem** | Python 🐍 | Core do projeto |
| **Framework** | Streamlit | Criação do Dashboard Interativo |
| **Manipulação** | Pandas | Tratamento e limpeza de dados (ETL) |
| **Visualização** | Plotly Express | Gráficos dinâmicos e mapas |
| **Machine Learning** | Scikit-learn | Algoritmos de K-Means e Random Forest |
| **Estatística** | Statsmodels | Cálculos de Regressão (OLS) |

---

## 🚀 Como Executar Localmente

1. **Clone o repositório**
```bash
   git clone [https://github.com/Vanessa-Matias/siob-data-science.git](https://github.com/Vanessa-Matias/siob-data-science.git)
   cd siob-data-science
 ```
2. **Instale as dependências**
```bash
   pip install -r requirements.txt
```
3. **Execute o Dashboard**
```bash
streamlit run analise_siob.py
````

---
## 🔗 Ecossistema SIOB

Este projeto faz parte de uma solução completa. Confira os outros repositórios:

* 💻 **Painel Operacional Web:** [Ver Repositório](https://github.com/Vanessa-Matias/cbmpe-siob-app)
* 📱 **Aplicativo Mobile:** [Ver Repositório](https://github.com/AgnesRibeiro/cbmpe-siob-app)

---

## 👩‍💻 Autora

**Vanessa Matias** *Desenvolvedora Fullstack & Analista de Dados em formação.* [LinkedIn](https://www.linkedin.com/in/vanessamatiasdev/) 

> Projeto desenvolvido para a disciplina de **Ciência de Dados** - Faculdade Senac Pernambuco (2025.2).
