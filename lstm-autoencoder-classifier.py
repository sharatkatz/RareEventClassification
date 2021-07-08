# -*- coding: utf-8 -*-
# -------------------------------------------
# Rare Event prediction with deep learning 
# model using LSTMs in tensorflow (2 ways - 
# one by using tensorflow slices and other
# by using numpy arrays as input to model)
# -------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from pylab import rcParams
import datetime

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from keras import optimizers, Sequential
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

from numpy.random import seed

seed(7)
tf.random.set_seed(11)

from sklearn.model_selection import train_test_split
import time
from functools import wraps


def timethis(func):
    """
    Decorator that reports the execution time.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            func.__name__,
            "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds),
        )
        return result

    return wrapper


class LSTMModel:
    def __init__(self, infile):
        try:
            raw_data = pd.read_csv(infile)
        except FileNotFoundError as Err:
            print("#" * 90)
            print(f"Check for the input file {infile}")
            print("#" * 90)
            sys.exit(1)

        self.features = [col for col in raw_data.columns if col not in ["time", "y"]]
        self.raw_data = raw_data.copy()
        self.lookback = 5
        self.batch_size = 128

    @staticmethod
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        """ A utility method to create a tf.data dataset from a Pandas Dataframe """
        dataframe = dataframe.copy()
        labels = dataframe.pop("y")
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds

    @staticmethod
    def find_windows(indf):
        """Prepare windows of data having
        one failure occurence in one window
        """
        counter = 1
        window = []
        y_series = indf["y"]
        for idx, elem in enumerate(y_series):
            if idx == 0:
                window.append(1)
            else:
                if y_series[idx] == 0 and y_series[idx - 1] == 1:
                    counter += 1
                    window.append(counter)
                else:
                    window.append(window[idx - 1])
        return window

    def find_modeling_vars(self, inDat):
        """Returns list of modeling vars"""
        min_dates = inDat.groupby("Window")["time"].min()
        max_dates = inDat.groupby("Window")["time"].max()
        left_table = pd.DataFrame()
        for id, from_date, to_date in zip(
            inDat.Window.unique().tolist(), min_dates, max_dates
        ):
            expand_ts = pd.date_range(from_date, to_date, freq="2min")
            expand_df = pd.DataFrame(expand_ts, columns=["time"])
            expand_df["Window"] = id
            left_table = pd.concat([left_table, expand_df], axis=0)

        to_impute = pd.merge(left_table, inDat, on=["Window", "time"], how="left")
        imputed_ts = LSTMModel.impute_ts(to_impute, "Window", "time", self.features)
        df = imputed_ts.copy()

        window_feat = []
        mod_vars = []
        for one_feature in self.features:
            for single_window in list(imputed_ts.Window.unique()):
                df = imputed_ts[imputed_ts["Window"] == single_window]
                try:
                    result = adfuller(df[one_feature])
                    if result[1] > 0.05:
                        window_feat.append([single_window, one_feature])
                        mod_vars.append(one_feature)
                        # print("{} - Series is not Stationary for Window - {}".format(one_feature, single_window))
                    else:
                        # print("{} - Series is Stationary for Window - {}".format(one_feature, single_window))
                        pass
                except ValueError:
                    print(
                        f"Test could not be performed for {one_feature} feature, for window {single_window}"
                    )

        return list(set(mod_vars))

    @staticmethod
    def impute_ts(indf, byvar, ordvar, impute_cols):
        # sort before imputation
        ts_dat = indf.sort_values(by=[byvar, ordvar])
        # forward pass
        ts_dat[impute_cols] = ts_dat.groupby(byvar)[impute_cols].fillna(method="ffill")
        # backward pass
        ts_dat[impute_cols] = ts_dat.groupby(byvar)[impute_cols].fillna(method="bfill")
        # Impute first with global mean and then with 0 for all missings case -
        # open to discussion
        ts_dat[impute_cols] = ts_dat[impute_cols].fillna(ts_dat[impute_cols].mean())
        ts_dat[impute_cols] = ts_dat[impute_cols].fillna(0)
        return ts_dat

    @staticmethod
    def getDuplicateColumns(df):
        """
        This function take a dataframe
        as a parameter and returning list
        of column names whose contents
        are duplicates.
        """

        # Create an empty set
        duplicateColumnNames = set()

        # Iterate through all the columns
        # of dataframe
        for x in range(df.shape[1]):

            # Take column at xth index.
            col = df.iloc[:, x]

            # Iterate through all the columns in
            # DataFrame from (x + 1)th index to
            # last index
            for y in range(x + 1, df.shape[1]):

                # Take column at yth index.
                otherCol = df.iloc[:, y]

                # Check if two columns at x & y
                # index are equal or not,
                # if equal then adding
                # to the set
                if col.equals(otherCol):
                    duplicateColumnNames.add(df.columns.values[y])

        # Return list of unique column names
        # whose contents are duplicates.
        return list(duplicateColumnNames)

    @staticmethod
    def lst_shift(list1, list2, lookback):
        assert len(list1) == len(list2), "Both lists should be of equal size"
        return np.vstack((list1[lookback: ], list2[: -lookback]))

    def data_shift(self, data_frame, lookback):
        window_recon = []
        data_frame = data_frame[["Window"] + ["y"] + self.features]
        df_mat = data_frame.values
        for window_num, single_window in enumerate(np.unique(df_mat[:, 0])):
            y = df_mat[df_mat[:, 0]==single_window, 1][lookback: ]
            feats_arr = []
            for feat_num, feat in enumerate(self.features):
                feat_arr = df_mat[df_mat[:, 0]==single_window, 2 + feat_num][ :-lookback]
                if feat_num == 0:
                    feats_arr = np.vstack((y, feat_arr))
                else:
                    feats_arr = np.vstack((feats_arr, feat_arr))
            if window_num == 0:
                window_recon = feats_arr
            else:
                window_recon = np.hstack((window_recon, feats_arr))
        out_dataframe = pd.DataFrame(window_recon.transpose(), columns=["y"] + self.features)
        out_dataframe["y"] = out_dataframe["y"].astype(int)
        return out_dataframe

    @staticmethod
    def temporalize(X, y, lookback):
        """
        LSTM is a bit more demanding than other models.
        Significant amount of time and attention goes in
        preparing the data that fits an LSTM.
        First, we will create the 3-dimensional
        arrays of shape: (samples x timesteps x features).
        Samples mean the number of data points.
        Timesteps is the number of time steps we look back
        at any time t to make a prediction. This is
        also referred to as lookback period. The features
        is the number of features the data has, in other words,
        the number of predictors in a multivariate data.
        """
        output_X = []
        output_y = []
        for i in range(len(X)-lookback-1):
            t = []
            for j in range(1,lookback+1):
                # Gather past records upto the lookback period
                t.append(X[i+j+1, :])
            output_X.append(t)
            output_y.append(y[i+lookback+1])
        return np.array(output_X), np.array(output_y)

    def data_for_modeling(self):
        self.raw_data["time"] = pd.to_datetime(self.raw_data["time"])
        # calculate time diff between two successive rows
        self.raw_data["time_diff"] = (
            self.raw_data["time"]
            .diff()
            .apply(lambda x: x / np.timedelta64(1, "m"))
            .fillna(0)
            .astype("int64")
        )
        # windows are built starting from a failure and back-tracking in time
        self.raw_data["Window"] = LSTMModel.find_windows(self.raw_data)
        self.raw_data["val"] = np.where(
            self.raw_data.time <= datetime.datetime(1999, 5, 22), "Train", "Test"
        )
        print(self.raw_data.groupby(["val", "y"]).size())
        # self.mod_vars = self.find_modeling_vars(raw_data)
        self.mod_vars = self.features
        print(f"[Number of variables that can be modeled are: {len(self.mod_vars)}]")
        # Shift data to accomodate the prediction window
        train_df = self.data_shift(self.raw_data.loc[self.raw_data.val == "Train", :], self.lookback)
        test_df = self.data_shift(self.raw_data.loc[self.raw_data.val == "Test", :], self.lookback)
        print(test_df.groupby(["y"]).size())
        print(train_df.groupby(["y"]).size())
        train_ds = LSTMModel.df_to_dataset(
           train_df,
           batch_size=self.batch_size,
        )
        test_ds = LSTMModel.df_to_dataset(
           test_df,
           batch_size=self.batch_size,
        )
        return train_df, test_df, train_ds, test_ds

    @staticmethod
    def get_normalization_layer(name, dataset):
        # Create a Normalization layer for our feature.
        normalizer = preprocessing.Normalization()
        # Prepare a Dataset that only yields our feature.
        feature_ds = dataset.map(lambda x, y: x[name])
        # Learn the statistics of the data.
        normalizer.adapt(feature_ds)
        return normalizer

    @staticmethod
    def normalize_numericfeatures(in_dat, numeric_columns):
        all_inputs = []
        encoded_features = []
        for header in numeric_columns:
            numeric_col = tf.keras.Input(shape=(1,), name=header)
            normalization_layer = LSTMModel.get_normalization_layer(header, in_dat)
            encoded_numeric_col = normalization_layer(numeric_col)
            all_inputs.append(numeric_col)
            encoded_features.append(encoded_numeric_col)
        return all_inputs, encoded_features

    @staticmethod
    def flatten(X):
        '''
        Flatten a 3D array.

        Input
        X            A 3D array for lstm, where the array is sample x timesteps x features.

        Output
        flattened_X  A 2D array, sample x features.
        '''
        flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1]-1), :]
        return(flattened_X)

    @staticmethod
    def scale(X, scaler):
        '''
        Scale 3D array.

        Inputs
        X            A 3D array for lstm, where the array is sample x timesteps x features.
        scaler       A scaler object, e.g., sklearn.preprocessing.StandardScaler, sklearn.preprocessing.normalize

        Output
        X            Scaled 3D array.
        '''
        for i in range(X.shape[0]):
            X[i, :, :] = scaler.transform(X[i, :, :])

        return X

    @staticmethod
    def build_model(train_ds, test_ds, all_inputs, encoded_features):
        all_features = tf.keras.layers.concatenate(encoded_features)
        x = tf.keras.layers.Dense(32, activation="relu")(all_features)
        x = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(all_inputs, output)
        METRICS = [
          keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        ]

        model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])

        print("Model Summary:")
        print(model.summary())
        model_history = model.fit(train_ds, epochs=10, validation_data=test_ds).history
        plt.plot(model_history['loss'], linewidth=2, label='Train Dataset')
        plt.plot(model_history['val_loss'], linewidth=2, label='Test Dataset')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

        plt.plot(model_history['accuracy'], linewidth=2, label='Train Dataset')
        plt.plot(model_history['val_accuracy'], linewidth=2, label='Test Dataset')
        plt.legend(loc='upper right')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.show()
        loss, accuracy = model.evaluate(test_ds)
        print("Accuracy", accuracy)
        print(model.metrics_names)
        return model

    @staticmethod
    def connectivity_graph(model):
        tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

    def auc(y_true, y_pred):
        auc = tf.metrics.auc(y_true, y_pred)[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc

    def evaluate_model(self, X_train, y_train, X_test, y_test, epochs, batch, lr):
        data_dim = 61
        timesteps = self.lookback
        nb_classes = 1

        model = Sequential()
        # Encoder
        model.add(LSTM(32, return_sequences=True, activation="relu",
            input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(32, activation="relu", return_sequences=True))  # returns a sequence of vectors of dimension 32
        # Decoder
        model.add(LSTM(16, activation='relu', return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(nb_classes)))
        print(model.summary())

        model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy']
        )

        tb = TensorBoard(log_dir='./logs',
                        histogram_freq=0,
                        write_graph=True,
                        write_images=True)

        lstm_autoencoder_history = model.fit(X_train, y_train,
                  validation_data = (X_test, y_test),
                  batch_size=64, epochs=epochs).history

        plt.plot(lstm_autoencoder_history['loss'], linewidth=2, label='Train Dataset')
        plt.plot(lstm_autoencoder_history['val_loss'], linewidth=2, label='Test Dataset')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

    @staticmethod
    def plot_roc(y_actual, y_pred, st_label):
        fpr, tpr, threshold = roc_curve(y_actual, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.title("Receiver Operating Characteristic")
        plt.plot(fpr, tpr, "b", label=st_label + " = %0.2f" % roc_auc)
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.show()

    @staticmethod
    def plot_confusion_matrix(
        cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
    ):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print("Confusion matrix, without normalization")

        print(cm)

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.show()


def main():
    import traceback

    try:
        tsa = LSTMModel("data.csv")
        # prepare modeling train and test 1s and 0s datasets
        train_df, test_df, train_ds, test_ds = tsa.data_for_modeling()
        #X_train, y_train = LSTMModel._temporalize(train_df[tsa.features].values, train_df.y.values, tsa.lookback)
        X_train, y_train = LSTMModel.temporalize(train_df[tsa.features].values, train_df.y.values, tsa.lookback)
        X_test, y_test = LSTMModel.temporalize(test_df[tsa.features].values, test_df.y.values, tsa.lookback)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)

        all_inputs, encoded_features = LSTMModel.normalize_numericfeatures(train_ds, tsa.features)
        model = LSTMModel.build_model(train_ds, test_ds, all_inputs, encoded_features)

        #model = tsa.evaluate_model(
        #        X_train, y_train, X_test, y_test, epochs=10, batch=tsa.batch_size, lr=0.0001
        #)

    except Exception as exp:
        print(exp.__class__, "ocurred in main..")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()

