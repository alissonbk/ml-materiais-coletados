import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#import plotly as px
import seaborn as sns
import matplotlib.pyplot as plt
import functions as fn
import numpy as np



def main():

    df = fn.pre_processing()

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
    x, y = df.iloc[:, :-1], df.iloc[:, [-1]]
    x_train, x_test, y_train, y_test = train_test_split(
                x, y.values.ravel(), test_size = 0.2, random_state=5)
    
    x_train = np.asarray(x_train).astype(np.int)
    #y_train = np.asarray(y_train).astype(np.int)
    y_train = np.asarray(fn.reduce_materiais_numbers(y_train)).astype(np.int)

    print(f"y_train: ${y_train}")

    model = tf.keras.models.Sequential([
        #tf.keras.layers.Dense(500, activation='relu', input_dim=10),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['categorical_accuracy'])
    
    model.fit(x_train, y_train, epochs=5)

    x_test = np.asarray(x_test).astype(np.int)
    y_test = np.asarray(fn.reduce_materiais_numbers(y_test)).astype(np.int)
    model.evaluate(x_test, y_test)
    


main()