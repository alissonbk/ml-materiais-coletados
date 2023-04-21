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

    print(df)

    #Encode dos campos string
    tipo_exploracao_encoder = preprocessing.LabelEncoder()
    materiais_encoder = preprocessing.LabelEncoder()
    newcastle_encoder = preprocessing.LabelEncoder()
    salmonella_encoder = preprocessing.LabelEncoder()
    #finalidade_encoder = preprocessing.LabelEncoder()
    df["tipo_exploracao"] = tipo_exploracao_encoder.fit_transform(df["tipo_exploracao"])
    df["materiais"] = materiais_encoder.fit_transform(df["materiais"].astype(str))# transforma o array para string
    df["vacina_newcastle"] = newcastle_encoder.fit_transform(df["vacina_newcastle"])
    df["vacina_salmonella"] = salmonella_encoder.fit_transform(df["vacina_salmonella"])
    #df["finalidade"] = finalidade_encoder.fit_transform(df["finalidade"])

    df["materiais"] = fn.transform_labels(df["materiais"])
    df = df.dropna()

    ##KNN
    print(df)
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
    knn = KNeighborsClassifier(n_neighbors=26, weights="distance", metric="manhattan")
    
    knn.fit(x_train, y_train)
    # Acuracia
    print(knn.score(x_test, y_test))

    # y_pred = knn.predict(x_test)

    # train = x_train;
    # train["materiais"] = y_train;
    # pred = x_test;
    # pred["materiais"] = y_pred;
    # print(train);
    
    #Plot
    df["materiais"] = materiais_encoder.inverse_transform(df["materiais"].astype(int))
    # train["materiais"] = materiais_encoder.inverse_transform(train["materiais"].astype(int))
    # train["tipo_exploracao"] = tipo_exploracao_encoder.inverse_transform(train["tipo_exploracao"])
    # pred["materiais"] = materiais_encoder.inverse_transform(pred["materiais"].astype(int))
    # pred["tipo_exploracao"] = tipo_exploracao_encoder.inverse_transform(pred["tipo_exploracao"])
    # df["finalidade"] = finalidade_encoder.inverse_transform(df["finalidade"].astype(int))
    df["vacina_newcastle"] = newcastle_encoder.inverse_transform(df["vacina_newcastle"].astype(int))
    df["vacina_salmonella"] = salmonella_encoder.inverse_transform(df["vacina_salmonella"].astype(int))
    df["tipo_exploracao"] = tipo_exploracao_encoder.inverse_transform(df["tipo_exploracao"].astype(int))
    
    sns.stripplot(x="tipo_exploracao", y="idade_lt6", hue="materiais", data=df)
    # plt.show()
    # sns.stripplot(x="vacina_salmonella", y="idade", hue="materiais", data=df, jitter=False, s=20, marker="D", linewidth=1, alpha=.1)
    # plt.show()
    # sns.stripplot(x="vacina_newcastle", y="idade", hue="materiais", data=df)
    # sns.scatterplot(x="tipo_exploracao", y="idade", hue="materiais", data=pred)
    #sns.pairplot(df,x_vars=["idade"],
    #y_vars=["vacina_newcastle", "vacina_salmonella", "tipo_exploracao"], hue='materiais')
    plt.show()

main()