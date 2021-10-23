import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import stats
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
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
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier


def load_dataset():
    # Importing the train and test dataset
    df_train = pd.read_csv("../train.csv")
    df_test = pd.read_csv("../test.csv")

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
    y_train_labeled = pd.get_dummies(y_train).to_numpy()
    y_test_labeled = pd.get_dummies(y_test).to_numpy()

    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train_labeled)
    # print(y_test_labeled.shape)
    return X_train, X_test, y_train_labeled, y_test_labeled


def train():
    xtrain, xtest, ytrain, ytest = load_dataset()
    xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))
    xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))

    timesteps, features, outputs = xtrain.shape[1], xtrain.shape[2], ytrain.shape[1]

    model = KerasClassifier(build_fn=model_rnn, verbose=0)

    param_grid = {
        'batch_size': [16, 32, 64, 128],
        'dropout_rate': [0.10, 0.20],
        'units': [16, 32, 64, 128],
        'layers': [1, 2]
    }
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        n_jobs=1,
                        cv=5, verbose=1)
    grid_result = grid.fit(timesteps, features)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print('Parameters:')
    for param, value in grid_result.best_params_.items():
        print('\t{}: {}'.format(param, value))

    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))

    best_model = grid_result.best_estimator_.model
    metric_names = best_model.metrics_names
    metric_values = best_model.evaluate(xtest, ytest)
    for metric, value in zip(metric_names, metric_values):
        print(metric, ': ', value)

def model_rnn(timesteps, features, outputs, units, layers, dropout_rate):
    # create model
    model = Sequential()
    # add model layers
    # 1. First Layer
    model.add(Bidirectional(LSTM(units, input_shape=(timesteps, features), return_sequences=True)))
    model.add(Dropout(dropout_rate))
    if layers == 1:
        # 2. Second layer
        model.add(Bidirectional(LSTM(units*2, return_sequences=True)))
        model.add(Dropout(dropout_rate))
    if layers == 2:
        # 2. Second layer
        model.add(Bidirectional(LSTM(units*2, return_sequences=True)))
        model.add(Dropout(dropout_rate))
        # 3. Second layer
        model.add(Bidirectional(LSTM(units*4, return_sequences=True)))
        model.add(Dropout(dropout_rate))

    model.add(Bidirectional(LSTM(units * 8, return_sequences=False)))

    model.add(Dropout(0.5))

    # 5. Output
    model.add(Dense(outputs, activation='softmax'))

    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


train()
