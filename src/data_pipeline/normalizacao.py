import pandas as pd
from sklearn import preprocessing
from pickle import dump, load
from pathlib import Path

# normalizacao dos dados
def normalizacao_df(df):
    print(" >> Normalizando dados << ")

    # separacao das colunas de targets
    colunas_targets = ['Diagnosis', 'Severity', 'Management']

    df_targets = df[colunas_targets] 
    df_features = df.drop(columns=colunas_targets)

    # normaliza colunas numericas
    colunas_numericas = df_features.select_dtypes(include=['float64', 'int64']).columns
    scaler = preprocessing.MinMaxScaler()
    df_features[colunas_numericas] = scaler.fit_transform(df_features[colunas_numericas])

    # aplica one-hot encoding nas colunas categoricas
    colunas_categoricas = df_features.select_dtypes(include=['object']).columns
    df_features = pd.get_dummies(df_features, columns=colunas_categoricas)

    # concatena features e targets
    df_processado = pd.concat([df_features, df_targets], axis=1)

    # salvar scaler na raiz
    root = Path(__file__).resolve().parent.parent
    scaler_path = root / 'models' / 'scaler.pkl'
    scaler_path.parent.mkdir(parents=True, exist_ok=True)  # Garante que o diretório exista
    dump(scaler, open(scaler_path, 'wb'))

    print(">> Normalização concluida <<")
    return df_processado
