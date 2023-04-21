from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import functions as fn




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


    ##SVM
    x, y = df.iloc[:, :-1], df.iloc[:, [-1]]
    x_train, x_test, y_train, y_test = train_test_split(
                x, y.values.ravel(), test_size = 0.2, random_state=5)
    
    clf = svm.SVC()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    
    #Plot
    # df["materiais"] = materiais_encoder.inverse_transform(df["materiais"].astype(int))
    # df["finalidade"] = finalidade_encoder.inverse_transform(df["finalidade"].astype(int))
    # df["vacina_newcastle"] = newcastle_encoder.inverse_transform(df["vacina_newcastle"].astype(int))
    # df["vacina_salmonella"] = salmonella_encoder.inverse_transform(df["vacina_salmonella"].astype(int))
    # df["tipo_exploracao"] = tipo_exploracao_encoder.inverse_transform(df["tipo_exploracao"].astype(int))
    # sns.boxplot(x="idade", y="tipo_exploracao", hue="materiais", data=df)
    # plt.show()

main()