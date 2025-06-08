from imblearn.over_sampling import SMOTE
import pandas as pd

def balanceamento_df(df, coluna_target):
    print("\n>> Balanceando dados << ")

    # separacao das colunas de targets
    X = df.drop(columns=['Diagnosis', 'Severity', 'Management'])
    y = df[coluna_target]

    # SMOTE para balanceamento
    smote = SMOTE(random_state=42)
    X_balanceado, y_balanceado = smote.fit_resample(X, y)

    # concatena features e targets
    df_balanceado = pd.concat([X_balanceado, y_balanceado], axis=1)

    print("> Balanceamento concluido <")
    return df_balanceado
