import numpy as np
import shap
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import arff
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from string import ascii_uppercase
import seaborn as sn
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier


def load_dataset(features):

    # Importing the train and test dataset
    df_train = pd.read_csv("../train.csv")
    df_test = pd.read_csv("../test.csv")

    # Maybe there are patterns in the data
    df_train = shuffle(df_train)
    df_test = shuffle(df_test)

    # Drop the index column
    df_train.reset_index(inplace=True)
    df_test.reset_index(inplace=True)
    df_train = df_train.drop(columns='index', axis=1)
    df_test = df_test.drop(columns='index', axis=1)

    # Create the train and label datasets
    X_train = df_train.drop(columns='Activity', axis=1)
    X_test = df_test.drop(columns='Activity', axis=1)

    X_train = X_train[features]
    X_test= X_test[features]
    # print(X_train.columns.tolist())
    # Scale our data so it has mean 0 and standard deviation of 1.
    scaler = StandardScaler()
    X_train_array = scaler.fit_transform(X_train)
    X_test_array = scaler.fit_transform(X_test)

    # Activity is our classification label
    y_train = df_train['Activity']
    y_test = df_test['Activity']

    # One hot encoding our labels for better representation of the classes for every sample
    # 0 means no, 1 means yes
    # y_train_labeled = pd.get_dummies(y_train).to_numpy()
    # y_test_labeled = pd.get_dummies(y_test).to_numpy()

    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train_labeled)
    # print(y_test_labeled.shape)
    return X_train_array, X_test_array, y_train, y_test, X_train


def train():
    xtrain, xtest, ytrain, ytest, X_train = load_dataset(0)
    callback = EarlyStopping(monitor='loss', patience=10)

    # With Shap
    model = model_dnn(0)


    y_train_labeled = pd.get_dummies(ytrain).to_numpy()
    y_test_labeled = pd.get_dummies(ytest).to_numpy()
    model.fit(xtrain, y_train_labeled, batch_size=64, epochs=200, validation_split=0.3,
                          verbose=0, callbacks=[callback])

    shap.initjs()
    background = xtrain[np.random.choice(xtrain.shape[0], 1000, replace=False)]
    explainer = shap.DeepExplainer(model=model, data=background)
    # shap values does perform fitting of the SHAP algorithm
    # based  on DeepLIFT using shapley sampling
    shap_values = explainer.shap_values(xtest[:100])
    shap.summary_plot(
        shap_values[0],
        xtest[:100],
        feature_names=X_train.columns,
        max_display=30,
        plot_type="bar",
        plot_size=(23, 9)
        )
    shap_sum = np.abs(shap_values).mean(axis=0)
    important_features = shap_sum.tolist()
    importance_df = pd.DataFrame([X_train.columns.tolist(), shap_sum.tolist()]).T

    importance_df.columns = ['column_name', 'shap_importance']

    importance_df = importance_df.sort_values('shap_importance', ascending=False)
    loss, accuracy = model.evaluate(xtest, y_test_labeled)
    print("\n Accuracy is : ", accuracy * 100)

    # for cross entropy loss
    print("\n Loss is : ", loss)

    return importance_df

def train2(features, x):
    xtrain, xtest, ytrain, ytest, X_train = load_dataset(features)
    kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    callback = EarlyStopping(monitor='loss', patience=10)

    accuracy_matrix = list()
    loss_matrix = list()
    ytest = pd.get_dummies(ytest).to_numpy()
    for k, (train_ind, test_ind) in enumerate(kf.split(xtrain, ytrain)):
        X_train_new, X_test_new = xtrain[train_ind], xtrain[test_ind]
        y_train_new, y_test_new = ytrain[train_ind], ytrain[test_ind]

        y_train_labeled = pd.get_dummies(y_train_new).to_numpy()
        y_test_labeled = pd.get_dummies(y_test_new).to_numpy()
        model = model_dnn(x)
        history = model.fit(X_train_new, y_train_labeled, batch_size=64, epochs=200,
                                validation_data=(X_test_new, y_test_labeled),
                                verbose=0, callbacks=[callback])

        loss, accuracy = model.evaluate(xtest, ytest)
        accuracy_matrix.append(accuracy)
        loss_matrix.append(loss)
    accuracy_mean = np.mean(accuracy_matrix)*100
    loss_mean  = np.mean(loss_matrix)
    print(f"\n Accuracy is  {accuracy_mean} with std {np.std(accuracy_matrix) * 100} ")

    # for cross entropy loss
    print(f"\n Loss is {loss_mean} with std {np.std(loss_matrix)}")

    # Plotting!
    plt.figure(0)
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy (train)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (test)')
    plt.title("Accuracy Metric")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Loss (train)')
    plt.plot(history.history['val_loss'], label='Loss (test)')
    plt.title("Loss Function")
    plt.ylabel("Values")
    plt.xlabel("Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    y_pred = model.predict(xtest)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(ytest, axis=1)

    columns = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Lying']
    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm, index=columns, columns=columns)
    sn.heatmap(cm, cmap='Oranges', annot=True)
    plt.tight_layout()
    plt.show()
    # print("F1 score: ", f1_score(y_true=ytest, y_pred=y_pred, average="macro"))
    return loss_mean, accuracy_mean

def model_dnn(x):
    # create model
    model = Sequential()
    # add model layers
    # 1. First Layer
    if x is 0:
        model.add(Dense(64, input_shape=(562,), activation='relu'))
    else:
        model.add(Dense(64, input_shape=(x,), activation='relu'))

    model.add(Dropout(0.1))
    # 2. Second layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))

    # 3. Third layer
    model.add(Dense(192, activation='relu'))
    model.add(Dropout(0.1))

    # 4. Fourth layer
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))

    # 5. Output
    model.add(Dense(6, activation='softmax'))

    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file="../DNN_Bestmodelplot.png", show_shapes=True, show_layer_names=True, rankdir="LR")
    return model


# importance = train()
# importance.to_csv("importance.csv", index=False)
# print(type(importance))
# train()
importance = pd.read_csv("../importance.csv")

num_of_top_features = [562, 200, 150, 100, 75, 50, 40, 30, 20, 10]
loss_features = []
acc_features = []
for x in num_of_top_features:
    features = importance.iloc[:x, 0]
    loss, accuracy = train2(features,x)
    print(f"\n Accuracy of the model for {x} top features is : ", accuracy)

    # for cross entropy loss
    print(f"\n Loss of the model for {x} top features is : ", loss)
    loss_features.append(loss)
    acc_features.append(accuracy)

x_pos = np.arange(len(num_of_top_features))
plt.bar(x_pos, loss_features)
plt.title("Loss per different amount of top features")
plt.ylabel("Loss")
plt.xlabel("Number of top features")
plt.xticks(x_pos, num_of_top_features)
plt.show()


plt.bar(x_pos, acc_features)
plt.title("Accuracy per different amount of top features")
plt.ylabel("Accuracy")
plt.xlabel("Number of top features")
plt.xticks(x_pos, num_of_top_features)
plt.show()