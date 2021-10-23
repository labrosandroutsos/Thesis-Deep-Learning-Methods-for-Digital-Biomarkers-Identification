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


def load_dataset(features):
    # Importing the train and test dataset
    df_train = pd.read_csv("../train.csv")
    df_test = pd.read_csv("../test.csv")

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


def train(features, x):
    xtrain, xtest, ytrain, ytest = load_dataset(features)
    xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))
    xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))

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
        if x == 562:
            steps, length = 2, 281
        elif x == 200:
            steps, length = 5, 40
        elif x == 150:
            steps, length = 5, 30
        elif x == 100:
            steps, length = 4, 25
        elif x == 75:
            steps, length = 3, 25
        elif x == 50:
            steps, length = 2, 25
        elif x == 40:
            steps, length = 4, 10
        elif x == 30:
            steps, length = 3, 10
        elif x == 20:
            steps, length = 2, 10
        elif x == 10:
            steps, length = 1, 10
        xtrain_new = X_train_new.reshape(X_train_new.shape[0], steps, length, features)
        xtest_new = X_test_new.reshape(X_test_new.shape[0], steps, length, features)
        model = model_cnnlstm(length, features)

        history = model.fit(xtrain_new, y_train_labeled, batch_size=64, epochs=200,
                            validation_data=(xtest_new, y_test_labeled),
                            verbose=0, callbacks=[callback])
        xtest_old = xtest.reshape(xtest.shape[0], steps, length, features)
        loss, accuracy = model.evaluate(xtest_old, ytest, verbose=1)
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

    y_pred = model.predict(xtest_old)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(ytest, axis=1)

    # columns = ['class %s' % (i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]
    columns = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Lying']
    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm, index=columns, columns=columns)
    sn.heatmap(cm, cmap='Oranges', annot=True)
    plt.tight_layout()
    plt.show()

    print(classification_report(y_test, y_pred, target_names=columns))
    return loss_mean, accuracy_mean


def model_cnnlstm(length, features):
    # create model
    model = Sequential()
    model.add(
        TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None, length, features)))
    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='relu')))
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


importance = pd.read_csv("../input/uci-har-features/importance.csv")

num_of_top_features = [562, 200, 150, 100, 75, 50, 40, 30, 20, 10]
loss_features = []
acc_features = []
for x in num_of_top_features:
    features = importance.iloc[:x, 0]
    loss, accuracy = train(features, x)
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