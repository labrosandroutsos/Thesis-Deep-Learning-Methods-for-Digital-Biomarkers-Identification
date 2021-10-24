import numpy as np
import tensorflow as tf
from scipy.io import arff
from scipy.stats import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow import keras
from imblearn.over_sampling import SMOTE

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, classification_report
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, LSTM, TimeDistributed, RepeatVector
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sn
from tensorflow.python.keras.models import Model, Sequential, load_model, save


def load_dataset(features):

    # Importing the train and test dataset
    df_train = pd.read_csv("../input/har-uci1/train.csv")
    df_test = pd.read_csv("../input/har-uci1/test.csv")


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
    X_test = X_test[features]
    # Scale our data so it has mean 0 and standard deviation of 1.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Activity is our classification label
    y_train = df_train['Activity']
    y_test = df_test['Activity']

    # One hot encoding our labels for better representation of the classes for every sample
    # 0 means no, 1 means yes
    # y_train_labeled = pd.get_dummies(y_train).to_numpy()
    # y_test_labeled = pd.get_dummies(y_test).to_numpy()

    return X_train, X_test, y_train, y_test


def train(features, x):
    xtrain, xtest, ytrain, ytest = load_dataset(features)

    callback = EarlyStopping(monitor='loss', patience=10)

    input, encoder, autoencoder = model_autoencoder_dense(x)
    history1 = autoencoder.fit(xtrain, xtrain, batch_size=32, epochs=200, validation_data=(xtest, xtest),
                        verbose=1, callbacks=[callback])

    X_train_encoded  = encoder.predict(xtrain)
    X_test_encoded = encoder.predict(xtest)
    print("Xtrain encoded", X_train_encoded.shape)
    print("Xtest encoded", X_test_encoded.shape)
    xtrain = np.reshape(X_train_encoded, (X_train_encoded.shape[0], X_train_encoded.shape[1], 1))
    xtest = np.reshape(X_test_encoded, (X_test_encoded.shape[0], X_test_encoded.shape[1], 1))
    accuracy_matrix = list()
    loss_matrix = list()
    ytest = pd.get_dummies(ytest).to_numpy()
    kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    for k, (train_ind, test_ind) in enumerate(kf.split(xtrain, ytrain)):
        X_train_new, X_test_new = xtrain[train_ind], xtrain[test_ind]
        y_train_new, y_test_new = ytrain[train_ind], ytrain[test_ind]

        y_train_labeled = pd.get_dummies(y_train_new).to_numpy()
        y_test_labeled = pd.get_dummies(y_test_new).to_numpy()
        model = model_cnn(x)

        history = model.fit(X_train_new, y_train_labeled, batch_size=64, epochs=200,
                            validation_data=(X_test_new, y_test_labeled),
                            verbose=0, callbacks=[callback])
        loss, accuracy = model.evaluate(xtest, ytest, verbose=1)
        accuracy_matrix.append(accuracy)
        loss_matrix.append(loss)

    accuracy_mean = np.mean(accuracy_matrix) * 100
    loss_mean = np.mean(loss_matrix)
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
    print(classification_report(y_test, y_pred, target_names=columns))
    return loss_mean, accuracy_mean


def train_wisdm():
    columns = ['user', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']

    # LOAD DATA
    data = pd.read_csv("../input/wisdm-dataset/WISDM_ar_v1.1_raw.txt", header=None, names=columns)
    data['z-axis'].replace({';': ''}, regex=True, inplace=True)
    data = data.dropna()

    data['activity'].value_counts().plot(kind='bar',
                                         title='Distribution of Human Activities for the WISDM dataset')
    plt.xticks(rotation=60)
    plt.xlabel("Activities")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    data_noact = data.drop(columns='activity', axis=1)

    # Activity is our classification label
    label_data = data['activity']

    # DATA PREPROCESSING
    data_convoluted = []
    labels = []
    SEGMENT_TIME_SIZE = 200
    TIME_STEP = 50

    # Slide a "SEGMENT_TIME_SIZE" wide window with a step size of "TIME_STEP"
    for i in range(0, len(data_noact) - SEGMENT_TIME_SIZE, TIME_STEP):
        x = data['x-axis'].values[i: i + SEGMENT_TIME_SIZE]
        y = data['y-axis'].values[i: i + SEGMENT_TIME_SIZE]
        z = data['z-axis'].values[i: i + SEGMENT_TIME_SIZE]
        data_convoluted.append([x, y, z])

        # Label for a data window is the label that appears most commonly
        label = stats.mode(label_data[i: i + SEGMENT_TIME_SIZE])[0][0]
        labels.append(label)

        #  Convert to numpy
    data_convoluted = np.asarray(data_convoluted, dtype=np.float32).transpose(0, 2, 1)
    # labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    labels = np.asarray((labels))

    print("data_convoluted after sliding window", data_convoluted.shape)
    print("labels after sliding window and get dummies", labels.shape)

    # SPLIT INTO TRAINING AND TEST SETS
    X_train, X_test, y_train, y_test = train_test_split(data_convoluted, labels, test_size=0.3,
                                                        random_state=13)
    oversample = SMOTE()
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)
    data, labeled = oversample.fit_resample(X_train, y_train)
    input, encoder, autoencoder = model_autoencoder_dense()
    callback = EarlyStopping(monitor='loss', patience=10)

    history1 = autoencoder.fit(data, data, batch_size=32, epochs=200, validation_data=(X_test, X_test),
                        verbose=1, callbacks=[callback])

    X_train_encoded  = encoder.predict(data)
    X_test_encoded = encoder.predict(X_test)
    print("Xtrain encoded", X_train_encoded.shape)
    print("Xtest encoded", X_test_encoded.shape)
    
    X_train = data.reshape(len(data), 200, 3)
    X_test = X_test_encoded.reshape(len(X_test_encoded), 200, 3)
    kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    accuracy_matrix = list()
    loss_matrix = list()

    y_test = pd.get_dummies(y_test).to_numpy()

    for k, (train_ind, test_ind) in enumerate(kf.split(X_train, labeled)):
        X_train_new, X_test_new = X_train[train_ind], X_train[test_ind]
        y_train_new, y_test_new = labeled[train_ind], labeled[test_ind]

        y_train_labeled = pd.get_dummies(y_train_new).to_numpy()
        y_test_labeled = pd.get_dummies(y_test_new).to_numpy()

        model = model_cnn()

        history = model.fit(X_train_new, y_train_labeled, batch_size=64, epochs=200, validation_data=(X_test_new, y_test_labeled),
                        callbacks=[callback],
                        verbose=1)

        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
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

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # columns = ['class %s' % (i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]
    columns = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm, index=columns, columns=columns)
    sn.heatmap(cm, cmap='Oranges', annot=True)
    plt.tight_layout()
    plt.show()
    print(classification_report(y_test, y_pred, target_names= columns))

def model_autoencoder_dense(x):
    # create model
    # 1. First Layer
    input = Input(shape=(x,))
    encoder = Dense(128, activation='relu')(input)
    encoder = Dense(64, activation='relu') (encoder)
    # bottleneck
    n_bottleneck = x /2
    bottleneck = Dense(n_bottleneck)(encoder)

    decoder = Dense(64, activation='relu')(bottleneck)
    decoder = Dense(128, activation='relu')(decoder)
    decoder = Dense(x, activation='linear') (decoder)

    encoded_model = Model(inputs=input, outputs=bottleneck)
    autoencoder = Model(inputs=input, outputs=decoder)
    autoencoder.compile(loss='mse', optimizer='adam')
    return input, encoded_model, autoencoder


def model_cnn(x):
    # create model
    x = int(x/2)
    global padding
    padding = 'valid'
    if x < 30:
        padding = 'same'
    model = Sequential()
    model.add(Conv1D(64, kernel_size=10, strides=1, padding=padding, input_shape=(x, 1), activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128,kernel_size=7, strides=1, padding=padding, activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))

    # 5. Output
    model.add(Dense(6, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    return model

importance = pd.read_csv("../importance.csv")
features = importance.iloc[:150, 0]
train(features, 150)
# train_wisdm()
