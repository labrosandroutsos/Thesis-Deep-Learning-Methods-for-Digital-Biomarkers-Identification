import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from scipy.io import arff
from scipy.stats import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, TimeDistributed, Flatten, Conv1D, MaxPooling1D
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from string import ascii_uppercase
import seaborn as sn
from tensorflow.python.keras.layers import AveragePooling1D, ConvLSTM2D, LSTM
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier


def load_dataset():

    # Importing the train and test dataset
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")

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
    # y_train_labeled = pd.get_dummies(y_train).to_numpy()
    # y_test_labeled = pd.get_dummies(y_test).to_numpy()

    return X_train, X_test, y_train, y_test


def train():
    xtrain, xtest, ytrain, ytest = load_dataset()
    xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))
    xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))


    # cv = StratifiedKFold()
    callback = EarlyStopping(monitor='loss', patience=10)
    # k = 1
    # for (train, test) in cv.split(xtrain, ytrain):
    #     X_train, X_test = xtrain[train, :], xtrain[test, :]
    #     y_train, y_test = ytrain[train], ytrain[test]
    #     print("{k} fold", X_train)
    #     print(y_train)
    # cv = StratifiedKFold()
    callback = EarlyStopping(monitor='loss', patience=10)
    kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    accuracy_matrix = list()
    loss_matrix = list()
    ytest = pd.get_dummies(ytest).to_numpy()
    for k, (train_ind, test_ind) in enumerate(kf.split(xtrain, ytrain)):
        X_train_new, X_test_new = xtrain[train_ind], xtrain[test_ind]
        y_train_new, y_test_new = ytrain[train_ind], ytrain[test_ind]

        y_train_labeled = pd.get_dummies(y_train_new).to_numpy()
        y_test_labeled = pd.get_dummies(y_test_new).to_numpy()

        timesteps, features, outputs = X_train_new.shape[1], X_train_new.shape[2], y_train_labeled.shape[1]
        steps, length = 2, 281
        xtrain_new = X_train_new.reshape(X_train_new.shape[0], steps, length, features)
        xtest_new = X_test_new.reshape(X_test_new.shape[0], steps, length, features)
        model = model_cnnlstm(steps,length, features)
    # model = KerasClassifier(build_fn=model_cnn, verbose=0)

    # param_grid = {
    #     'batch_size': [16, 32, 64, 128],
    #     'epochs': [100, 200],
    #     'dropout_rate': [0.10, 0.25, 0.3],
    #     'pool_type': ['max', 'average'],
    #     'filters': [32, 64, 128],
    #     'dense_neurons': [32, 64, 128],
    #     'layers': [1,2,3,4]
    # }
        history = model.fit(xtrain_new, y_train_labeled, batch_size=64, epochs=200, validation_data=(xtest_new, y_test_labeled),
                        verbose=1, callbacks=[callback])
        xtest_old = xtest.reshape(xtest.shape[0], steps, 1, length, features)
        loss, accuracy = model.evaluate(xtest_old, ytest)
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

    # columns = ['class %s' % (i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]
    columns = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Lying']
    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm, index=columns, columns=columns)
    sn.heatmap(cm, cmap='Oranges', annot=True)
    plt.tight_layout()
    plt.show()

    print(classification_report(y_test, y_pred, target_names= columns))

def train_wisdm():
    columns = ['user', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']

    # LOAD DATA
    data = pd.read_csv("../WISDM_ar_v1.1_raw.txt", header=None, names=columns)
    data['z-axis'].replace({';': ''}, regex=True, inplace=True)
    data = data.dropna()

    # DATA PREPROCESSING
    data_convoluted = []
    labels = []
    SEGMENT_TIME_SIZE = 200
    TIME_STEP = 50

    # Slide a "SEGMENT_TIME_SIZE" wide window with a step size of "TIME_STEP"
    for i in range(0, len(data) - SEGMENT_TIME_SIZE, TIME_STEP):
        x = data['x-axis'].values[i: i + SEGMENT_TIME_SIZE]
        y = data['y-axis'].values[i: i + SEGMENT_TIME_SIZE]
        z = data['z-axis'].values[i: i + SEGMENT_TIME_SIZE]
        data_convoluted.append([x, y, z])

        # Label for a data window is the label that appears most commonly
        label = stats.mode(data['activity'][i: i + SEGMENT_TIME_SIZE])[0][0]
        labels.append(label)


    data_convoluted = np.asarray(data_convoluted, dtype=np.float32).transpose(0, 2, 1)

    # SPLIT INTO TRAINING AND TEST SETS
    X_train, X_test, y_train, y_test = train_test_split(data_convoluted, labels, test_size=0.3,
                                                        random_state=13)

    kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    accuracy_matrix = list()
    loss_matrix = list()

    y_test = pd.get_dummies(y_test).to_numpy()
    callback = EarlyStopping(monitor='loss', patience=10)

    for k, (train_ind, test_ind) in enumerate(kf.split(X_train, y_train)):
        y_train = pd.get_dummies(y_train).to_numpy()
        X_train_new, X_test_new = X_train[train_ind], X_train[test_ind]
        y_train_new, y_test_new = y_train[train_ind], y_train[test_ind]
        model = model_cnnlstm()

        history = model.fit(X_train_new, y_train_new, batch_size=64, epochs=200, validation_data=(X_test_new, y_test_new),
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

def model_cnnlstm(steps, length, features):
    # create model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None, length, features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    # 4. Fully Connected layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))

    # 5. Output
    model.add(Dense(6, activation='softmax'))

    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# train_wisdm()
train()