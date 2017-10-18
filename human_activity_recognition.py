import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input


class HAR:
    def __init__(self, data_folder="data", train_file="train.csv", test_file="test.csv"):
        self.data_folder = data_folder
        self.train_file = train_file
        self.test_file = test_file

        self.train_data = None
        self.test_data = None
        self.n_features = None  # all columns except subject,Activity

        self.flag_encoded = False  # Set this True after encoding based on train data
        self.label_encoder = LabelEncoder()
        # ?? Should onehot dtype be changed to int?
        self.onehot_encoder = OneHotEncoder(sparse=False)

    def load_train_data(self):
        with open(os.path.join(self.data_folder, self.train_file), 'r') as fd:
            self.train_data = pd.read_csv(fd)
            self.n_features = self.train_data.shape[1] - 2

    def load_test_data(self):
        with open(os.path.join(self.data_folder, self.test_file), 'r') as fd:
            self.test_data = pd.read_csv(fd)

    def prepare_data_for_lstm(self, data, n_timestep=5):
        """Preparing data_x,data_y for input to LSTM

            Parameters
            ----------
            data : pandas dataframe (created by loading train/test data)

            Returns
            -------
            (data_x, data_y) : tuple (data point for LSTM)
        """
        n_sample = data.shape[0]
        print "sample size: {0} : feature size: {1}".format(n_sample, data.shape[1])
        data_x = []
        row_i = 0
        row_i_array = []
        # create datapoints using n_timestep
        while row_i < n_sample:
            # case #1: row_i,..,row_i+n_timestep-1 : belongs to same (subject,Activity)
            # case #2: (subject,Activity) changes before row_i+n_timestep:
            #           Check if we can create datapoint by shifting row_i back i.e. creating datapoint which overlap
            #            with previous datapoint

            # Get the labels i.e. (subject,Activity)
            cur_subject = data.ix[row_i, "subject"]
            cur_activity = data.ix[row_i, "Activity"]

            # Check for case #1
            row_j = row_i + 1
            # If case #1 satisfies then row_j - row_i = n_timestep i.e. datapoint [row_i,row_j)
            flag_sequence = True
            while (row_j < (row_i+n_timestep)) and (row_j < n_sample):
                if (data.ix[row_j, "subject"] == cur_subject) and (data.ix[row_j, "Activity"] == cur_activity):
                    row_j += 1
                else:
                    flag_sequence = False
                    break

            assert row_j-row_i <= n_timestep, "Error: while computing row_j"
            # TODO another assert to match row_j with flag_sequence

            if (row_j - row_i < n_timestep) and (row_j > row_i+1):
                # Check for case #2
                # 2nd condition i.e. row_j > row_i+1 is required to avoid getting the same datapoint as previous one
                #       This can be modified to control how much overlap we can allow
                row_k = row_i - 1

                while (row_k >= (row_j - n_timestep)) and (row_k >= 0):
                    if (data.ix[row_k, "subject"] == cur_subject) and (data.ix[row_k, "Activity"] == cur_activity):
                        row_k -= 1
                    else:
                        break

                # check if it can form datapoint
                if (row_j - row_k) == (n_timestep+1):
                    row_i = row_k + 1
                    flag_sequence = True

            if flag_sequence is True:
                # Now create the datapoint for n_timestamp: [row_i,row_j) as explained by Adam Sypniewski
                cur_datapoint = []
                for row_index in range(row_i, row_j):
                    cur_timestamp_x_data = np.array(data.ix[row_index, data.columns.difference(["subject","Activity"])])
                    cur_datapoint.append(cur_timestamp_x_data)

                data_x.append(cur_datapoint)
                row_i_array.append(row_i)

            # assigning starting row for next datapoint
            row_i = row_j

        data_x = np.array(data_x)
        # Now create the data_y
        # https://stackoverflow.com/questions/33385238/how-to-convert-pandas-single-column-data-frame-to-series-or-numpy-vector
        y_str_values = self.train_data['Activity'].values[row_i_array]
        if self.flag_encoded is False:
            y_int_encoded = self.label_encoder.fit_transform(y_str_values)
            y_int_encoded = y_int_encoded.reshape(len(y_str_values), 1)
            data_y = self.onehot_encoder.fit_transform(y_int_encoded)
            self.flag_encoded = True
        else:
            y_int_encoded = self.label_encoder.transform(y_str_values)
            y_int_encoded = y_int_encoded.reshape(len(y_str_values), 1)
            data_y = self.onehot_encoder.transform(y_int_encoded)

        return data_x, data_y

    def train_model(self, n_timestep=5, nb_epoch=5):
        """Train the LSTM model for many-to-one
        """
        # Create each datapoint with 5 timesteps (say).
        # We are doing this as each (subject,Activity) combination don't have the same number of timesteps

        # create data_x, data_y for training
        # similarly create data_x, data_y for testing

        train_data_x, train_data_y = self.prepare_data_for_lstm(self.train_data, n_timestep)
        n_class = len(train_data_y[0])
        test_data_x, test_data_y = self.prepare_data_for_lstm(self.test_data, n_timestep)

        model = Sequential()
        model.add(LSTM(output_dim=100, return_sequences=False, input_shape=(n_timestep, self.n_features)))
        model.add(Dense(output_dim=n_class, activation="sigmoid"))
        # https://keras.io/losses/
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        print(model.summary())
        model.fit(x=train_data_x, y=train_data_y, nb_epoch=nb_epoch)
        scores = model.evaluate(test_data_x, test_data_y)
        print("\nAccuracy: %.2f%%" % (scores[1] * 100))

if __name__ == "__main__":
    har_obj = HAR()
    har_obj.load_train_data()
    har_obj.load_test_data()
    har_obj.train_model(n_timestep=10, nb_epoch=5)


"""
References:
    Followed the explanation in these forums:
    https://datascience.stackexchange.com/questions/17024/rnns-with-multiple-features
    https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

    One-hot encoding:
    https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    step #1: encode string to integer
    step #2: one-hot encode integer to one-hot
    Mentions that sparse encoding might not be suitable for Keras

    https://stackoverflow.com/questions/33385238/how-to-convert-pandas-single-column-data-frame-to-series-or-numpy-vector
"""