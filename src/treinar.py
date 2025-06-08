import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier
from pickle import dump
from pathlib import Path

def treinar_modelo(df, coluna_target, nome_modelo):
    print(f" >> Treinando modelo {nome_modelo} << ")

    # Remove apenas a coluna de target específica
    X = df.drop(columns=[coluna_target])
    y = df[coluna_target]

    # modelo random forest
    modelo = RandomForestClassifier(random_state=42)
    # parametros para o grid search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    # grid search
    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X, y)

    # treina modelo final com os THE BESTS param_grid
    modelo_final = RandomForestClassifier(**grid_search.best_params_, random_state=42)
    modelo_final.fit(X, y)

    # cross validation
    cv_results = cross_validate(
        modelo_final, X, y, cv=10, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    )

    print(f' >> Acuracy: {cv_results["test_accuracy"].mean()}')
    print(f' >> Precision: {cv_results["test_precision_macro"].mean()}')
    print(f' >> Recall: {cv_results["test_recall_macro"].mean()}')
    print(f' >> F1: {cv_results["test_f1_macro"].mean()}')

    # salvar modelo com nome específico na raiz
    root = Path(__file__).resolve().parent.parent
    model_path = root / 'models' / f'modelo_{nome_modelo}.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)  # Garante que o diretório exista
    dump(modelo_final, open(model_path, 'wb'))

    print(f">> Treinamento do modelo {nome_modelo} concluido <<")
    return modelo_final