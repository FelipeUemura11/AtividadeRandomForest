from data_pipeline.processamento import import_df, preprocessamento_df
from data_pipeline.normalizacao import normalizacao_df
from data_pipeline.balanceamento import balanceamento_df
from treinar import treinar_modelo
from pathlib import Path

def criacao_diretorios():
    root = Path(__file__).resolve().parent.parent
    directories = [root / 'data', root / 'models']
    for directory in directories:
        directory.mkdir(exist_ok=True)

def executar_pipeline_de_treinamento():
    print("Iniciando pipeline de treinamento...")
    
    # 1. importação e pre-processamento
    print("\n> Importando e pré-processando dados...")
    dados_raw = import_df()
    dados_processados = preprocessamento_df(dados_raw)
    
    # 2. normalização
    print("\n> Fazendo a Normalizando dados...")
    dados_normalizados = normalizacao_df(dados_processados)
    print("> Dados processados e normalizados <")
    
    # 3. treinamento dos modelos
    # 3.1 diagnosis
    print("\n> Treinando modelo para [Diagnosis]...")
    dados_diagnosis = dados_normalizados.copy()
    dados_diagnosis = balanceamento_df(dados_diagnosis, 'Diagnosis')
    treinar_modelo(dados_diagnosis, 'Diagnosis', 'diagnosis')
    
    # 3.2 severity (apenas casos de apendicite)
    print("\n> Treinando modelo para [Severity]...")
    dados_severity = dados_normalizados[dados_normalizados['Diagnosis'] == 'appendicitis'].copy()
    dados_severity = balanceamento_df(dados_severity, 'Severity')
    treinar_modelo(dados_severity, 'Severity', 'severity')
    
    # 3.3 Modelo para Management (apenas casos de apendicite)
    print("\n> Treinando modelo para [Management]...")
    dados_management = dados_normalizados[dados_normalizados['Diagnosis'] == 'appendicitis'].copy()
    dados_management = balanceamento_df(dados_management, 'Management')
    treinar_modelo(dados_management, 'Management', 'management')
    
    print("\n> Pipeline de treinamento concluído com sucesso!")

def main():
    # criar diretorios modelos e data
    criacao_diretorios()
    
    while True:
        print('====================================')
        print(" >> Pediatric Uemura's Diagnosis << ")
        print(" [1] Treinar modelos                ")
        print(" [2] Inferir modelo                 ")
        print(" [0] Sair                           ")
        print('====================================')

        try:
            opc = int(input("Digite sua opcao: "))
            
            if opc == 1:
                executar_pipeline_de_treinamento()
            elif opc == 2:
                print("inferir")
            elif opc == 0:
                print("Saindo do programa...")
                break
            else:
                print("Opcao invalida, tente novamente")
        except ValueError:
            print("Por favor, digite um numero valido")

if __name__ == "__main__":
    main()