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

# Cria grupos de idade
def idade_in_groups(df):
    idade_lt6 = [];
    idade_lt15 = [];
    idade_lt40 = [];
    idade_lt50 = [];
    idade_lt100 = [];
    idade_lt150 = [];
    idade_mt150 = [];
    
    for idx, d in enumerate(df["idade"]):
        if d < 6:
            idade_lt6.append(1);
            idade_lt15.append(0);
            idade_lt40.append(0);
            idade_lt50.append(0);
            idade_lt100.append(0);
            idade_lt150.append(0);
            idade_mt150.append(0);
        elif d < 15:
            idade_lt6.append(0);
            idade_lt15.append(1);
            idade_lt40.append(0);
            idade_lt50.append(0);
            idade_lt100.append(0);
            idade_lt150.append(0);
            idade_mt150.append(0);
        elif d < 40:
            idade_lt6.append(0);
            idade_lt15.append(0);
            idade_lt40.append(1);
            idade_lt50.append(0);
            idade_lt100.append(0);
            idade_lt150.append(0);
            idade_mt150.append(0);
        elif d < 50:
            idade_lt6.append(0);
            idade_lt15.append(0);
            idade_lt40.append(0);
            idade_lt50.append(1);
            idade_lt100.append(0);
            idade_lt150.append(0);
            idade_mt150.append(0);
        elif d < 100:
            idade_lt6.append(0);
            idade_lt15.append(0);
            idade_lt40.append(0);
            idade_lt50.append(0);
            idade_lt100.append(1);
            idade_lt150.append(0);
            idade_mt150.append(0);
        elif d < 150:
            idade_lt6.append(0);
            idade_lt15.append(0);
            idade_lt40.append(0);
            idade_lt50.append(0);
            idade_lt100.append(0);
            idade_lt150.append(1);
            idade_mt150.append(0);
        elif d >= 150:
            idade_lt6.append(0);
            idade_lt15.append(0);
            idade_lt40.append(0);
            idade_lt50.append(0);
            idade_lt100.append(0);
            idade_lt150.append(0);
            idade_mt150.append(1);
    
    materiais = df["materiais"] # to keep label in last col
    df.drop("materiais", axis=1, inplace=True )
    df["idade_lt6"] = idade_lt6
    df["idade_lt15"] = idade_lt15
    df["idade_lt40"] = idade_lt40
    df["idade_lt50"] = idade_lt50
    df["idade_lt100"] = idade_lt100
    df["idade_lt150"] = idade_lt150
    df["idade_mt150"] = idade_mt150
    df.drop('idade', axis=1, inplace=True)
    df["materiais"] = materiais
    return df

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

def pre_processing():
    df = pd.read_csv('materiais2.csv')
    df = transform_idade(df)
    df = transform_str_to_array(df)
    df = drop_unnecessary_features(df)
    df = idade_in_groups(df)

    return df