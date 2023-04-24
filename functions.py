import pandas as pd
import numpy as np
#import plotly as px


def transform_idade(df):
    for idx, d in enumerate(df["idade"]):
        if(d.__contains__('semana')):
            d = d.replace("semanas", "").strip()
            d = d.replace("semana", "").strip()
            df.loc[idx, "idade"] = int(d) * 7
        if(d.__contains__('dia')):
            d = d.replace("dias", "").strip()
            df.loc[idx, "idade"] = int(d.replace("dia", "").strip())
        if(d.__contains__('seamans')):
            df.loc[idx, "idade"] = int(d.replace("seamans", "").strip()) * 7

    return df

# def transform_idade_semanas(df):
#     for idx, d in enumerate(df["idade"]):
#         if(d.__contains__('semana')):
#             d = d.replace("semanas", "").strip()
#             d = d.replace("semana", "").strip()
#             df.loc[idx, "idade"] = int(d)
#         if(d.__contains__('dia')):
#             d = d.replace("dias", "").strip()
#             df.loc[idx, "idade"] = int(d.replace("dia", "").strip()) / 7
#         if(d.__contains__('seamans')):
#             df.loc[idx, "idade"] = int(d.replace("seamans", "").strip())

#     return df

def transform_str_to_array(df):
    df["materiais"] = df["materiais_coletados_array"]
    df["materiais"] = df["materiais"].astype(object)

    for idx, d in enumerate(df["materiais_coletados_array"]):
        e = d.replace("{", "").replace("}", "").replace("\"", "").strip()
        df.at[idx, "materiais"] = list(np.array(e.split(",")))

    return df

def flatten(l):
    return [item for sublist in l for item in sublist]


def show_most_similars(items):
    items = items.tolist()
    x = []
    duplicateds_freq = {}
    selecteds = []


    for d in items:
        if(d not in x):
            x.append(d)
    
    for i in x:
        duplicateds_freq[i] = items.count(i)

    for i in duplicateds_freq:
        if duplicateds_freq[i] > 100:
            print(i)

    print(duplicateds_freq)

## Reduz a quantia de labels para as que ocorrem mais de NUM_OCORRENCIAS
def reduce_labels(items):
    #NUM_OCORRENCIAS = 220 #5 classes
    NUM_OCORRENCIAS = 500
    items = items.tolist()
    x = []
    duplicateds_freq = {}
    selecteds = []

    for d in items: 
        if(d not in x):
            x.append(d)
    
    for i in x:
        duplicateds_freq[i] = items.count(i)

    for i in duplicateds_freq:
        if duplicateds_freq[i] > NUM_OCORRENCIAS:
            selecteds.append(i)
    
    print(f'Numero de classes: {set(selecteds).__len__()}')
    return selecteds

def transform_labels(items):
    reduced_labels = reduce_labels(items)
    selecteds = []

    for i in items:
        if(reduced_labels.__contains__(i)):
            selecteds.append(i)
        else:
            selecteds.append(np.nan)
    

    return selecteds

def drop_unnecessary_features(df):
    df.drop("materiais_coletados_array", inplace=True, axis=1)
    #df.drop("idcolheita", inplace=True, axis=1)

    df.drop("nro_aves", inplace=True, axis=1) #Numero de aves é uma feature com baixa correlação
    df.drop("exploracao", inplace=True, axis=1) #Dataset com monitorias oficiais possui apenas reprodução
    df.drop("segmento", inplace=True, axis=1) #Dataset com monitorias oficiais possui apenas AVES_GENETICA
    df.drop("finalidade", inplace=True, axis=1)

    #df.drop("vacina_newcastle", inplace=True, axis=1)
    #df.drop("vacina_salmonella", inplace=True, axis=1)
    #df.drop("tipo_exploracao", inplace=True, axis=1)

    return df

# Para cada grupo de material o label encoder gerou um número no caso as 3 classes ficaram com um número acima 
# de 100, para a camada de output seria interessante deixar 0 1 2
def reduce_materiais_numbers(materiais):
    new_materiais = []

    #FIXME static numbers for the encoded arrays
    for m in materiais:
        if m == 145:
            new_materiais.append(0)
        if m == 179:
            new_materiais.append(1)
        if m == 194:
            new_materiais.append(2)

    return new_materiais



def pre_processing():
    df = pd.read_csv('materiais2.csv')
    df = transform_idade(df)
    df = transform_str_to_array(df)
    df = drop_unnecessary_features(df)

    return df