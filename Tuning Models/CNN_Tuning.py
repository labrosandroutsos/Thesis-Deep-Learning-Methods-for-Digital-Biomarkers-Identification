import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import arff
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from string import ascii_uppercase
import seaborn as sn
from tensorflow.python.keras.layers import AveragePooling1D
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier


def load_dataset():

    # Importing the train and test dataset
    df_train = pd.read_csv("../train.csv")
    df_test = pd.read_csv("../test.csv")
    #
    # sn.displot(df_train, x="Activities", shrink=.8)
    # plt.tight_layout()
    # plt.show()
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

    model = KerasClassifier(build_fn=model_cnn, verbose=0)

    param_grid = {
        'batch_size': [16, 32, 64, 128],
        'dropout_rate': [0.10, 0.25, 0.3],
        'pool_type': ['max', 'average'],
        'filters': [32, 64, 128],
        'dense_neurons': [32, 64, 128],
        'layers': [1, 2, 3]
    }
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        n_jobs=1,
                        cv=5, verbose=1)
    grid_result = grid.fit(xtrain, ytrain)
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


def model_cnn(pool_type, dropout_rate, filters, dense_neurons, layers):
    # create model
    model = Sequential()
    # add model layers
    # 1. First Convolutional Layer
    model.add(Conv1D(filters, kernel_size=10, strides=1, input_shape=(562, 1), activation='relu'))
    # dropout layer
    model.add(Dropout(dropout_rate))
    # pooling layer
    if pool_type == 'max':

        model.add(MaxPooling1D(pool_size=2))

    else:
        model.add(AveragePooling1D(pool_size=2))

    if layers == 1:
        # 2. Second Convolutional layer
        model.add(Conv1D(filters * 2, kernel_size=7, strides=1, activation='relu'))
        # dropout layer
        model.add(Dropout(dropout_rate))
        # pooling layer
        if pool_type == 'max':
            model.add(MaxPooling1D(pool_size=2))
        else:
            model.add(AveragePooling1D(pool_size=2))
    elif layers == 2:
        # 3. Third Convolutional layer
        model.add(Conv1D(filters * 3, kernel_size=5, strides=1, activation='relu'))
        # dropout layer
        model.add(Dropout(dropout_rate))
        # pooling layer
        if pool_type == 'max':
            model.add(MaxPooling1D(pool_size=2))
        else:
            model.add(AveragePooling1D(pool_size=2))
        # 4. Fourth Convolutional layer
        model.add(Conv1D(filters * 4, kernel_size=3, strides=1, activation='relu'))
        # dropout layer
        model.add(Dropout(dropout_rate))
        # pooling layer
        if pool_type == 'max':
            model.add(MaxPooling1D(pool_size=2))
        else:
            model.add(AveragePooling1D(pool_size=2))

    elif layers == 3:
        # 3. Third Convolutional layer
        model.add(Conv1D(filters * 3, kernel_size=5, strides=1, activation='relu'))
        # dropout layer
        model.add(Dropout(dropout_rate))
        # pooling layer
        if pool_type == 'max':
            model.add(MaxPooling1D(pool_size=2))
        else:
            model.add(AveragePooling1D(pool_size=2))
        # 4. Fourth Convolutional layer
        model.add(Conv1D(filters * 4, kernel_size=3, strides=1, activation='relu'))
        # dropout layer
        model.add(Dropout(dropout_rate))
        # pooling layer
        if pool_type == 'max':
            model.add(MaxPooling1D(pool_size=2))
        else:
            model.add(AveragePooling1D(pool_size=2))
        # 5. Fifth Convolutional layer
        model.add(Conv1D(filters * 5, kernel_size=1, strides=1, activation='relu'))
        # dropout layer
        model.add(Dropout(dropout_rate))
        # pooling layer
        if pool_type == 'max':
            model.add(MaxPooling1D(pool_size=2))
        else:
            model.add(AveragePooling1D(pool_size=2))


    model.add(Flatten())

    # 4. Fully Connected layer
    model.add(Dense(dense_neurons, activation='relu'))
    # 5. Output
    model.add(Dense(6, activation='softmax'))

    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

train()