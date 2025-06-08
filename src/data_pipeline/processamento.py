import pandas as pd
from ucimlrepo import fetch_ucirepo
# importacao do dataset
def import_df():
    print(">> Importando dados do UCI << ")
    df = fetch_ucirepo(id=938)

    X = df.data.features
    y = df.data.targets

    df = pd.concat([X, y], axis=1)
    print("> Dataset importado!! < ")
    return df

# preprocessamento dos dados
def preprocessamento_df(df):
    print("\n>> Processando dados << ")

    colunas_remover = [ 
        'Segmented_Neutrophils', 'Appendix_Wall_Layers', 'Target_Sign', 'Appendicolith',
        'Perfusion', 'Perforation', 'Surrounding_Tissue_Reaction', 'Appendicular_Abscess',
        'Abscess_Location', 'Pathological_Lymph_Nodes', 'Lymph_Nodes_Location',
        'Bowel_Wall_Thickening', 'Conglomerate_of_Bowel_Loops', 'Ileus', 'Coprostasis',
        'Meteorism', 'Enteritis', 'Gynecological_Findings'
    ]
    # remocao das colunas irrelevantes
    df = df.drop(columns=colunas_remover)

    colunas_categoricas = [
        'Sex', 'Management', 'Severity', 'Neutrophilia', 'Ketones_in_Urine', 'Stool', 
        'Contralateral_Rebound_Tenderness', 'Coughing_Pain', 'Nausea', 'Loss_of_Appetite', 
        'RBC_in_Urine', 'WBC_in_Urine', 'Dysuria', 'Peritonitis', 'Psoas_Sign', 
        'Ipsilateral_Rebound_Tenderness', 'US_Performed', 'Free_Fluids', 'Diagnosis', 
        'Appendix_on_US', 'Migratory_Pain', 'Lower_Right_Abd_Pain'
    ]

    colunas_numericas = [
        'Age', 'BMI', 'Height', 'Weight', 'Length_of_Stay', 'Appendix_Diameter', 
        'Body_Temperature', 'WBC_Count', 'Neutrophil_Percentage', 'RBC_Count', 
        'Hemoglobin', 'RDW', 'Thrombocyte_Count', 'CRP', 'Alvarado_Score', 
        'Paedriatic_Appendicitis_Score'
    ]
    # preenchimento das colunas categoricas com a moda
    for coluna in colunas_categoricas:
        df[coluna] = df[coluna].fillna(df[coluna].mode()[0])

    # preenchimento das colunas numericas com a mediana
    for coluna in colunas_numericas:
        df[coluna] = df[coluna].fillna(df[coluna].median())

    print("> Processamento concluído < ")
    return df
