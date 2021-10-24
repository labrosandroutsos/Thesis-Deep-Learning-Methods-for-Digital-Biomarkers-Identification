# DeepInsight library: https://github.com/alok-ai-lab/DeepInsight
from pyDeepInsight import ImageTransformer, LogScaler
from imblearn.over_sampling import SMOTE

# Data (image) augmentation
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.stats import stats
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from matplotlib import pyplot as plt, ticker
from scipy.io import arff
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, AveragePooling2D
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
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.utils.vis_utils import plot_model


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

    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train_labeled)
    # print(y_test_labeled.shape)
    return X_train, X_test, y_train, y_test


def train(features, x):
    xtrain, xtest, ytrain, ytest = preprocessData(features)

    callback = EarlyStopping(monitor='loss', patience=5)
    kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    accuracy_matrix = list()
    loss_matrix = list()
    ytest = pd.get_dummies(ytest).to_numpy()
    for k, (train_ind, test_ind) in enumerate(kf.split(xtrain, ytrain)):
        X_train_new, X_test_new = xtrain[train_ind], xtrain[test_ind]
        y_train_new, y_test_new = ytrain[train_ind], ytrain[test_ind]

        y_train_labeled = pd.get_dummies(y_train_new).to_numpy()
        y_test_labeled = pd.get_dummies(y_test_new).to_numpy()
        model = model_cnn()

        plot_model(model, to_file="2D-CNN_modelplot.png", show_shapes=True, show_layer_names=True)
        history = model.fit(X_train_new, y_train_labeled, batch_size=64, epochs=200,
                            validation_data=(X_test_new, y_test_labeled),
                            verbose=0, callbacks=[callback])
        loss, accuracy = model.evaluate(xtest, ytest)
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


def train_wisdm():
    columns = ['user', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']

    # LOAD DATA
    data = pd.read_csv("../WISDM_ar_v1.1_raw.txt", header=None, names=columns)
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
    labels = np.asarray((labels))
    # SPLIT INTO TRAINING AND TEST SETS
    X_train, X_test, y_train, y_test = train_test_split(data_convoluted, labels, test_size=0.3,
                                                        random_state=13)
    oversample = SMOTE()
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)
    data, labeled = oversample.fit_resample(X_train, y_train)

    data, X_test = preprocessData(data, X_test, labeled, y_test)

    X_train = data
    kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    accuracy_matrix = list()
    loss_matrix = list()
    y_test = pd.get_dummies(y_test).to_numpy()
    for k, (train_ind, test_ind) in enumerate(kf.split(X_train, labeled)):
        if k==1 :
            X_train_new, X_test_new = X_train[train_ind], X_train[test_ind]
            y_train_new, y_test_new = labeled[train_ind], labeled[test_ind]

            y_train_labeled = pd.get_dummies(y_train_new).to_numpy()
            y_test_labeled = pd.get_dummies(y_test_new).to_numpy()

            model = model_cnn()
            callback = EarlyStopping(monitor='loss', patience=10)

            history = model.fit(X_train_new, y_train_labeled, batch_size=64, epochs=200,
                            validation_data=(X_test_new, y_test_labeled), callbacks=[callback],
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
    print(classification_report(y_test, y_pred, target_names=columns))


# transforms the samples of data into 2-dimensional images

# this is crucial to input the data into the convolutional neural network
def convertToImageData(X_train_norm, X_test_norm, y_train, y_test):
    it = ImageTransformer(feature_extractor='kpca', pixels=33, random_state=None)

    # fit and transform the data into image matrices
    X_train_img = it.fit_transform(X_train_norm)
    X_test_img = it.fit_transform(X_test_norm)

    return X_train_img, X_test_img


# processes the data to be used in the convolutional neural network model
def preprocessData(X_train, X_test, y_train, y_test):
    X_train_img, X_test_img = convertToImageData(X_train, X_test, y_train, y_test)
    print("diastaseis eikona", X_train_img.shape)

    return X_train_img, X_test_img


def model_cnn():
    # create model
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu', padding='same',
                     input_shape=(33, 33, 3)))
    model.add(MaxPooling2D((2, 2), (1, 1)))
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), (1, 1)))
    model.add(Conv2D(64, (7, 7), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), (1, 1)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    # 5. Output
    model.add(Dense(6, activation='softmax'))

    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# importance = pd.read_csv("importance.csv")
# features = importance.iloc[:150, 0]
# train(features, 150)
train_wisdm()


