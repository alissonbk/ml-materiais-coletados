from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#import plotly as px
import seaborn as sns
import matplotlib.pyplot as plt
import functions as fn
import numpy as np
from sklearn.model_selection import GridSearchCV



def main():

    df = fn.pre_processing()

    #Encode dos campos string
    tipo_exploracao_encoder = preprocessing.LabelEncoder()
    materiais_encoder = preprocessing.LabelEncoder()
    newcastle_encoder = preprocessing.LabelEncoder()
    salmonella_encoder = preprocessing.LabelEncoder()
    finalidade_encoder = preprocessing.LabelEncoder()
    df["tipo_exploracao"] = tipo_exploracao_encoder.fit_transform(df["tipo_exploracao"])
    df["materiais"] = materiais_encoder.fit_transform(df["materiais"].astype(str))# transforma o array para string
    df["vacina_newcastle"] = newcastle_encoder.fit_transform(df["vacina_newcastle"])
    df["vacina_salmonella"] = salmonella_encoder.fit_transform(df["vacina_salmonella"])
    #df["finalidade"] = finalidade_encoder.fit_transform(df["finalidade"])

    df["materiais"] = fn.transform_labels(df["materiais"])
    df = df.dropna()
    print('------------------------\n', df)

    ##KNN
    x, y = df.iloc[:, :-1], df.iloc[:, [-1]]
    x_train, x_test, y_train, y_test = train_test_split(
                x, y.values.ravel(), test_size = 0.2, random_state=5)
    
    #GridSearch
    # grid_params = { 'n_neighbors' : [2,5,7,9,11,13,15,20,23,26,28,35,40,45,50],
    #                'weights' : ['uniform','distance'],
    #                'metric' : ['minkowski','euclidean','manhattan', 
    #                            'cosine', 'jaccard', 'hamming']}
    # gs = GridSearchCV(KNeighborsClassifier(), 
    #                   grid_params, verbose = 1, cv=3, n_jobs = -1)

    # gfit = gs.fit(x_train, y_train)
    # print(f"Best Score: {gfit.best_score_} \n Best Params = {gfit.best_params_}")
    #Best Params = {'metric': 'cosine', 'n_neighbors': 28, 'weights': 'distance'}

    #79.882
    knn = KNeighborsClassifier()
    
    knn.fit(x_train, y_train)
    # Acuracia
    print(knn.score(x_test, y_test))


main()