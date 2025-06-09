import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
import os

# Diretórios
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CSV_PATH = DATA_DIR / "pacientes_inferidos.csv"

# Carrega o scaler
try:
    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = load(f)
except Exception as e:
    print(f"ERRO ao carregar o scaler: {str(e)}")
    scaler = None

# Colunas exatamente como definidas no processamento
COLUNAS_NUMERICAS = [
    'Age', 'BMI', 'Height', 'Weight', 'Length_of_Stay', 'Alvarado_Score',
    'Paedriatic_Appendicitis_Score', 'Appendix_Diameter', 'Body_Temperature',
    'WBC_Count', 'Neutrophil_Percentage', 'RBC_Count', 'Hemoglobin', 'RDW',
    'Thrombocyte_Count', 'CRP'
]

# Colunas que serão perguntadas ao usuário (removendo as colunas de target)
COLUNAS_CATEGORICAS = [
    'Sex', 'Appendix_on_US', 'Migratory_Pain', 'Lower_Right_Abd_Pain',
    'Contralateral_Rebound_Tenderness', 'Coughing_Pain', 'Nausea',
    'Loss_of_Appetite', 'Neutrophilia', 'Ketones_in_Urine', 'RBC_in_Urine',
    'WBC_in_Urine', 'Dysuria', 'Stool', 'Peritonitis', 'Psoas_Sign',
    'Ipsilateral_Rebound_Tenderness', 'US_Performed', 'Free_Fluids'
]

# Colunas de target que serão previstas
COLUNAS_TARGET = ['Diagnosis', 'Severity', 'Management']

# Lista de colunas que o MODELO espera (com one-hot encoding)
ENCODED_FEATURES_NAMES = [
    'Age', 'BMI', 'Height', 'Weight', 'Length_of_Stay', 'Alvarado_Score',
    'Paedriatic_Appendicitis_Score', 'Appendix_Diameter', 'Body_Temperature',
    'WBC_Count', 'Neutrophil_Percentage', 'RBC_Count', 'Hemoglobin', 'RDW',
    'Thrombocyte_Count', 'CRP', 'Sex_female', 'Sex_male', 'Appendix_on_US_no',
    'Appendix_on_US_yes', 'Migratory_Pain_no', 'Migratory_Pain_yes',
    'Lower_Right_Abd_Pain_no', 'Lower_Right_Abd_Pain_yes',
    'Contralateral_Rebound_Tenderness_no', 'Contralateral_Rebound_Tenderness_yes',
    'Coughing_Pain_no', 'Coughing_Pain_yes', 'Nausea_no', 'Nausea_yes',
    'Loss_of_Appetite_no', 'Loss_of_Appetite_yes', 'Neutrophilia_no',
    'Neutrophilia_yes', 'Ketones_in_Urine_+', 'Ketones_in_Urine_++',
    'Ketones_in_Urine_+++', 'Ketones_in_Urine_no', 'RBC_in_Urine_+',
    'RBC_in_Urine_++', 'RBC_in_Urine_+++', 'RBC_in_Urine_no', 'WBC_in_Urine_+',
    'WBC_in_Urine_++', 'WBC_in_Urine_+++', 'WBC_in_Urine_no', 'Dysuria_no',
    'Dysuria_yes', 'Stool_constipation', 'Stool_constipation, diarrhea',
    'Stool_diarrhea', 'Stool_normal', 'Peritonitis_generalized',
    'Peritonitis_local', 'Peritonitis_no', 'Psoas_Sign_no', 'Psoas_Sign_yes',
    'Ipsilateral_Rebound_Tenderness_no', 'Ipsilateral_Rebound_Tenderness_yes',
    'US_Performed_no', 'US_Performed_yes', 'Free_Fluids_no', 'Free_Fluids_yes'
]

def verificar_diretorios():
    """
    Verifica se os diretórios necessários existem.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def normalizar_dados(paciente):
    """
    Normaliza os dados do paciente usando o scaler.
    """
    if scaler is None:
        print("ERRO: Scaler não foi carregado corretamente!")
        return None
        
    try:
        # Separa dados numéricos e categóricos
        colunas_numericas = paciente[COLUNAS_NUMERICAS]
        colunas_categoricas = paciente[COLUNAS_CATEGORICAS]

        # Normaliza dados numéricos
        paciente_num_normalizados = scaler.transform(colunas_numericas)
        paciente_num_normalizados = pd.DataFrame(paciente_num_normalizados, columns=COLUNAS_NUMERICAS)

        # Aplica One-Hot Encoding nos dados categóricos
        paciente_cat_normalizados = pd.get_dummies(colunas_categoricas, dtype='int')
        
        # Junta os dataframes
        paciente_normalizado = pd.concat([paciente_num_normalizados, paciente_cat_normalizados], axis=1)

        # Garante que todas as colunas do modelo estejam presentes
        paciente_final = pd.DataFrame(columns=ENCODED_FEATURES_NAMES)
        paciente_final = pd.concat([paciente_final, paciente_normalizado], ignore_index=True).fillna(0)

        return paciente_final[ENCODED_FEATURES_NAMES]  # Retorna na ordem correta e com todas as colunas
        
    except Exception as e:
        print(f"ERRO ao normalizar dados numéricos: {str(e)}")
        return None

def inferir_target(paciente_normalizado, target):
    """
    Realiza a inferência para um target específico.
    """
    try:
        # Carrega o modelo específico para o target
        modelo_path = MODELS_DIR / f"modelo_{target}.pkl"
        with open(modelo_path, "rb") as f:
            modelo = load(f)

        # Realiza a predição
        proba = modelo.predict_proba(paciente_normalizado)
        return proba

    except Exception as e:
        print(f"ERRO ao inferir {target}: {str(e)}")
        return None

def salvar_inferencia_csv(dados):
    """
    Salva os resultados da inferência em um arquivo CSV.
    """
    verificar_diretorios()
    
    arquivo_existe = CSV_PATH.exists()
    
    try:
        with open(CSV_PATH, 'a', newline='') as f:
            writer = pd.DataFrame([dados]).to_csv(f, header=not arquivo_existe, index=False)
    except Exception as e:
        print(f"ERRO ao salvar resultados: {str(e)}")

def inferir_paciente(paciente):
    """
    Realiza a inferência completa para um paciente.
    """
    # Verifica se o DataFrame tem todas as colunas necessárias
    colunas_faltantes = set(COLUNAS_CATEGORICAS) - set(paciente.columns)
    if colunas_faltantes:
        print(f"ERRO: Colunas categóricas faltando: {colunas_faltantes}")
        return

    # Normaliza dados do paciente
    paciente_normalizado = normalizar_dados(paciente)
    if paciente_normalizado is None:
        return

    # Realiza inferência para cada target
    resultados = {}
    for target in COLUNAS_TARGET:
        proba = inferir_target(paciente_normalizado, target)
        if proba is None:
            return
        resultados[target] = proba[0]

    # Prepara dados para salvar
    dados_completos = paciente.iloc[0].to_dict()
    dados_completos.update({
        'Diagnosis': 'appendicitis' if resultados['Diagnosis'][0] > 0.5 else 'no appendicitis',
        'Severity': 'complicated' if resultados['Severity'][0] > 0.5 else 'uncomplicated',
        'Management': 'conservative' if resultados['Management'][0] > 0.5 else 'primary surgical'
    })

    # Salva resultados
    salvar_inferencia_csv(dados_completos)
    print(f"\n> Inferência salva em '{CSV_PATH.name}' <")

def validar_valor_numerico(valor, coluna):
    """
    Valida valores numéricos com base na coluna.
    """
    if coluna == 'Age':
        return 0 <= valor <= 18  # Idade pediátrica
    elif coluna == 'BMI':
        return 10 <= valor <= 40  # IMC razoável para crianças
    elif coluna == 'Height':
        return 50 <= valor <= 200  # Altura em cm
    elif coluna == 'Weight':
        return 5 <= valor <= 100  # Peso em kg
    elif coluna == 'Length_of_Stay':
        return 0 <= valor <= 30  # Dias de internação
    elif coluna == 'Appendix_Diameter':
        return 0 <= valor <= 20  # Diâmetro em mm
    elif coluna == 'Body_Temperature':
        return 35 <= valor <= 42  # Temperatura em Celsius
    elif coluna == 'WBC_Count':
        return 1000 <= valor <= 50000  # Contagem de leucócitos
    elif coluna == 'Neutrophil_Percentage':
        return 0 <= valor <= 100  # Porcentagem de neutrófilos
    elif coluna == 'RBC_Count':
        return 1 <= valor <= 10  # Contagem de hemácias
    elif coluna == 'Hemoglobin':
        return 5 <= valor <= 20  # Hemoglobina
    elif coluna == 'RDW':
        return 10 <= valor <= 20  # RDW
    elif coluna == 'Thrombocyte_Count':
        return 50000 <= valor <= 500000  # Plaquetas
    elif coluna == 'CRP':
        return 0 <= valor <= 200  # PCR
    elif coluna == 'Alvarado_Score':
        return 0 <= valor <= 10  # Escore de Alvarado
    elif coluna == 'Paedriatic_Appendicitis_Score':
        return 0 <= valor <= 10  # Escore pediátrico
    return True

def obter_input_usuario():
    """
    obtem os dados do paciente atraves de input do usuario.
    """
    print("\n> ENTRADA DE DADOS DO PACIENTE <")
    
    dados = {}
    
    # Coleta dados numéricos
    for coluna in COLUNAS_NUMERICAS:
        while True:
            try:
                valor = float(input(f"{coluna}: "))
                if validar_valor_numerico(valor, coluna):
                    dados[coluna] = valor
                    break
                else:
                    print(f"Valor fora do intervalo aceitável para {coluna}")
            except ValueError:
                print("Por favor, insira um número válido.")
    
    # Coleta dados categóricos
    for coluna in COLUNAS_CATEGORICAS:
        if coluna == 'Sex':
            valor = input(f"{coluna} (M/F): ").upper()
            while valor not in ['M', 'F']:
                print("Por favor, insira 'M' ou 'F'.")
                valor = input(f"{coluna} (M/F): ").upper()
        elif coluna in ['Ketones_in_Urine', 'RBC_in_Urine', 'WBC_in_Urine']:
            valor = input(f"{coluna} (no/+/++/+++): ").lower()
            while valor not in ['no', '+', '++', '+++']:
                print("Por favor, insira 'no', '+', '++' ou '+++'.")
                valor = input(f"{coluna} (no/+/++/+++): ").lower()
        elif coluna == 'Stool':
            valor = input(f"{coluna} (normal/constipation/diarrhea/constipation, diarrhea): ").lower()
            while valor not in ['normal', 'constipation', 'diarrhea', 'constipation, diarrhea']:
                print("Por favor, insira uma opção válida.")
                valor = input(f"{coluna} (normal/constipation/diarrhea/constipation, diarrhea): ").lower()
        elif coluna == 'Peritonitis':
            valor = input(f"{coluna} (no/local/generalized): ").lower()
            while valor not in ['no', 'local', 'generalized']:
                print("Por favor, insira 'no', 'local' ou 'generalized'.")
                valor = input(f"{coluna} (no/local/generalized): ").lower()
        else:
            valor = input(f"{coluna} (yes/no): ").lower()
            while valor not in ['yes', 'no']:
                print("Por favor, insira 'yes' ou 'no'.")
                valor = input(f"{coluna} (yes/no): ").lower()
        
        dados[coluna] = valor
    
    return pd.DataFrame([dados])

def main():
    """
    Função principal que coordena o fluxo de inferência.
    """
    # Verifica diretórios
    if not verificar_diretorios():
        return

    # Coleta dados do paciente
    paciente = obter_input_usuario()
    if paciente is None:
        return

    inferir_paciente(paciente)

if __name__ == "__main__":
    main()
