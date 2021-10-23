import pandas as pd
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from sklearn.utils import shuffle
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier


def load_dataset():

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

    # Scale our data so it has mean 0 and standard deviation of 1.
    scaler = StandardScaler()
    X_train_array = scaler.fit_transform(X_train)
    X_test_array = scaler.fit_transform(X_test)

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
    return X_train_array, X_test_array, y_train_labeled, y_test_labeled, X_train


def train():
    xtrain, xtest, ytrain, ytest, X_train = load_dataset()

    model = KerasClassifier(build_fn=model_dnn, verbose=0)
    param_grid = {
        'batch_size': [16, 32, 64, 128],
        'dropout_rate': [0.10, 0.25],
        'dense_neurons': [32, 64, 128],
        'layers': [1, 2, 3, 4]
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


def model_dnn(dense_neurons, dropout_rate, layers):
    # create model
    model = Sequential()
    # add model layers
    # 1. First Layer

    model.add(Dense(dense_neurons, input_shape=(562,), activation='relu'))

    model.add(Dropout(dropout_rate))
    if layers == 1:
        # 2. Second layer
        model.add(Dense(dense_neurons*2, activation='relu'))
        model.add(Dropout(dropout_rate))
    elif layers == 2:
        # 2. Second layer
        model.add(Dense(dense_neurons * 2, activation='relu'))
        model.add(Dropout(dropout_rate))
        # 3. Third layer
        model.add(Dense(dense_neurons*3, activation='relu'))
        model.add(Dropout(dropout_rate))
    elif layers == 3:
        # 2. Second layer
        model.add(Dense(dense_neurons * 2, activation='relu'))
        model.add(Dropout(dropout_rate))
        # 3. Third layer
        model.add(Dense(dense_neurons * 3, activation='relu'))
        model.add(Dropout(dropout_rate))
        # 4. Fourth layer
        model.add(Dense(dense_neurons*4, activation='relu'))
        model.add(Dropout(dropout_rate))

    elif layers == 4:
        # 2. Second layer
        model.add(Dense(dense_neurons * 2, activation='relu'))
        model.add(Dropout(dropout_rate))
        # 3. Third layer
        model.add(Dense(dense_neurons * 3, activation='relu'))
        model.add(Dropout(dropout_rate))
        # 4. Fourth layer
        model.add(Dense(dense_neurons*4, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(dense_neurons * 5, activation='relu'))

    # 5. Output
    model.add(Dense(6, activation='softmax'))

    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

train()
