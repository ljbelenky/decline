import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, Dense



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

    for i in range(10):
        initial = np.random.uniform(1000)
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
    data['target'] = ys

    data.transpose().plot()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1])

    model = LR().fit(X_train, y_train)
    print(model.score(X_train, y_train))
    print(model.score(X_test, y_test))

    # plt.scatter(y_train, model.predict(X_train))
    # plt.scatter(y_test, model.predict(X_test))
    # plt.show()

    model_m = Sequential()
    model_m.add(Conv1D(100, 10, activation='relu', input_shape=(None, X_train.shape[1])))
    model_m.add(Conv1D(100, 10, activation='relu'))
    model_m.add(Conv1D(160, 10, activation='relu'))
    model_m.add(Conv1D(160, 10, activation='relu'))
    model_m.add(Dense(1, activation='linear'))
    print(model_m.summary())
    model_m.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model_m.fit(X_train,y_train)
    