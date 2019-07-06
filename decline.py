# https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Reshape



class Decline_Generator():
    '''This class creates synthetic gas production data with dropouts, low point, high points and noise.
    Return data is a pandas dataframe with columns "date" and "production'''
    def __init__(self, initial = 100, drops = {'num':0,'max_length':0}, noise = 0, highpoints = (0,0), lowpoints = (0,0), decline_rate = .996, flush = 8):
        n = 256
        # dates = pd.date_range(start = '2018-01-01', periods = n)
        production = initial
        base_production = []
        for _ in range(n):
            base_production.append(production)
            production*=decline_rate
        data = pd.DataFrame({'production':base_production})

        self.target = production

        data.production += np.random.random(size = n)*noise*initial/100

        for i in range(highpoints[0]):
            index = np.random.randint(n)
            data.production.iat[index] += highpoints[1]*initial/100

        for i in range(lowpoints[0]):
            index = np.random.randint(n)
            data.production.iat[index] -= lowpoints[1]*initial/100

        for i in range(drops['num']):
            start = np.random.randint(n)
            length = np.random.randint(drops['max_length'])
            data.production.iloc[start:start+length] = 0  
            
        data.production += np.random.random(size = n)*noise-noise/2
        data.production = data.production.map(lambda x:max(x,0))
        self.data = data.values.flatten()

    def __call__(self):
        return self.data, self.target

if __name__ == '__main__':

    Xs, ys = [], []
    batch_size = 100
    for i in range(batch_size):
        initial = np.random.uniform(1000)
        initial = 1000
        decline_rate = np.random.uniform(.995, .9999)
        drops = {'num':np.random.randint(15), 'max_length':np.random.randint(1,20)}
        noise = np.random.uniform(16)
        highpoints = np.random.randint(20), np.random.uniform(50)
        lowpoints = np.random.randint(20), np.random.uniform(40)

        dg = Decline_Generator(initial = initial, drops = drops, noise = noise, highpoints = highpoints, lowpoints = lowpoints, decline_rate = decline_rate)
        X,y = dg()
        Xs.append(X)
        ys.append(y)

    data = pd.DataFrame(Xs)
    target = ys

    # data.transpose().plot()
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(data, target)
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis =2)
    # y_train = y_train.values.reshape(len(y_train), 1)
    # y_test = y_test.values.reshape(len(y_test), 1)


    # model = Sequential()
    # # model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
    # model.add(Conv1D(100, 10, activation='relu', input_shape=(256,1)))
    # model.add(Conv1D(100, 10, activation='relu'))
    # model.add(Conv1D(160, 10, activation='relu'))
    # model.add(Conv1D(160, 10, activation='relu'))
    # model.add(Dense(1, activation='linear'))
    # print(model.summary())
    # model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # model.fit(X_train,y_train)
    

    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Embedding
    from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

    seq_length = 256

    model = Sequential()
    model.add(Conv1D(100, 16, activation='relu', input_shape=(seq_length, 1)))
    model.add(Conv1D(50, 8, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(50, 4, activation='relu'))
    model.add(Conv1D(25, 2, activation='relu'))
    model.add(GlobalAveragePooling1D())
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error',
                optimizer='adam')

    model.fit(X_train, y_train, epochs=5000, batch_size = batch_size)
    score = model.evaluate(X_test, y_test)
    print(score**.5)