# Sistema de Diagnóstico Pediátrico

Sistema de machine learning para auxiliar no diagnóstico de apendicite em pacientes pediátricos, utilizando dados clínicos e laboratoriais.

## Estrutura do Projeto

```
Pediatric_Diagnosis/
├── data/                  # Dados de treinamento e inferência
├── models/               # Modelos treinados e scaler
├── src/                  # Código fonte
│   ├── data_pipeline/    # Scripts de processamento de dados
│   ├── main.py          # Interface principal do sistema
│   ├── treinar.py       # Script de treinamento do modelo
│   └── inferencia.py    # Script para fazer inferências
├── requirements.txt      # Dependências do projeto
└── README.md
```

## Requisitos

- Python 3.8+
- pandas
- scikit-learn
- imbalanced-learn
- ucimlrepo
- numpy

Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como Usar

1. **Interface Principal**:
```bash
python src/main.py

```
```

## Funcionalidades

- Treinamento de modelo Random Forest para diagnóstico de apendicite
- Inferência em novos casos usando dados clínicos
- Normalização automática dos dados (StandardScaler)
- One-Hot Encoding para variáveis categóricas
- Suporte a múltiplas features clínicas e laboratoriais

## Features Utilizadas

- Dados demográficos (idade, sexo, IMC)
- Sinais clínicos (dor, temperatura, etc.)
- Exames laboratoriais (hemograma, urina)
- Exames de imagem (ultrassom)
- Scores clínicos (Alvarado, Pediatric Appendicitis)
