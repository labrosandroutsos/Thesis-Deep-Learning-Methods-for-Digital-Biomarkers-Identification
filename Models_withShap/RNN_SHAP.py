from string import ascii_uppercase

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex
from collections import Counter

from mlxtend.plotting import plot_confusion_matrix
from scipy.stats import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn
from tensorflow.python.keras.layers import GRU
from imblearn.over_sampling import SMOTE


def load_dataset(features):
    # Importing the train and test dataset
    df_train = pd.read_csv("../train.csv")
    df_test = pd.read_csv("../test.csv")

    # Create the train and label datasets
    X_train = df_train.drop(columns='Activity', axis=1)
    X_test = df_test.drop(columns='Activity', axis=1)
    X_train = X_train[features]
    X_test= X_test[features]
    # Scale our data so it has mean 0 and standard deviation of 1.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test
                                  )
    # Activity is our classification label
    y_train = df_train['Activity']
    y_test = df_test['Activity']

    # One hot encoding our labels for better representation of the classes for every sample
    # 0 means no, 1 means yes
#     y_train_labeled = pd.get_dummies(y_train).to_numpy()
#     y_test_labeled = pd.get_dummies(y_test).to_numpy()

    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train_labeled)
    # print(y_test_labeled.shape)
    return X_train, X_test, y_train, y_test


def train(features, x):
    xtrain, xtest, ytrain, ytest = load_dataset(features)
    xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))
    xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))

    kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    accuracy_matrix = list()
    loss_matrix = list()
    ytest = pd.get_dummies(ytest).to_numpy()

    for k, (train_ind, test_ind) in enumerate(kf.split(xtrain, ytrain)):
        if k==1:
            X_train_new, X_test_new = xtrain[train_ind], xtrain[test_ind]
            y_train_new, y_test_new = ytrain[train_ind], ytrain[test_ind]

            y_train_labeled = pd.get_dummies(y_train_new).to_numpy()
            y_test_labeled = pd.get_dummies(y_test_new).to_numpy()
            timesteps, features, outputs = X_train_new.shape[1], X_train_new.shape[2], y_train_labeled.shape[1]
            model = model_rnn(timesteps, features, outputs)
            callback = EarlyStopping(monitor='loss', patience=10)

            history = model.fit(X_train_new, y_train_labeled, batch_size=64, epochs=200,
                                    validation_data=(X_test_new, y_test_labeled), callbacks=[callback],
                                    verbose=0)

            loss, accuracy = model.evaluate(xtest, ytest, verbose=1)
            accuracy_matrix.append(accuracy)
            loss_matrix.append(loss)

    print(f"\n Accuracy is  {np.mean(accuracy_matrix) * 100} with std {np.std(accuracy_matrix) * 100} ")

    # for cross entropy loss
    print(f"\n Loss is {np.mean(loss_matrix)} with std {np.std(loss_matrix)}")

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

    print(classification_report(y_test, y_pred, target_names=columns))
    accuracy_mean = np.mean(accuracy_matrix)*100
    loss_mean = np.mean(loss_matrix)
    return loss_mean, accuracy_mean

def model_rnn(timesteps, features, outputs):
    # create model
    model = Sequential()
    # add model layers
    # 1. First Layer
    model.add(Bidirectional(LSTM(64, input_shape=(timesteps, features), return_sequences=True)))
    model.add(Dropout(0.2))
    # 2. Second layer
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.2))
    # 3. Second layer
    model.add(Bidirectional(LSTM(256, return_sequences=False)))

    model.add(Dropout(0.5))

    # 5. Output
    model.add(Dense(outputs, activation='softmax'))

    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


importance = pd.read_csv("../input/uci-har-features/importance.csv")
# train()
num_of_top_features = [562, 200, 150, 100, 75, 50, 40, 30, 20, 10]
loss_features = []
acc_features = []
for x in num_of_top_features:
    features = importance.iloc[:x, 0]
    loss, accuracy = train(features,x)
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