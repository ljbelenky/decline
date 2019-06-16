import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression as LR

class Decline_Generator():
    '''This class creates synthetic gas production data with dropouts, low point, high points and noise.
    Return data is a pandas dataframe with columns "date" and "production'''
    def __init__(self, n = 366, drops = (0,0), 
    noise = 0, highpoints = (0,0), lowpoints = (0,0), decline_rate = .996):
        dates = pd.date_range(start = '2018-01-01', end = '2019-01-01', periods = n)
        x = 100
        base_production = []
        for _ in range(n):
            base_production.append(x)
            x*=decline_rate
        data = pd.DataFrame({'date':dates,'production':base_production})

        data.production += np.random.random(size = n)*noise

        for i in range(highpoints[0]):
            index = np.random.randint(n)
            data.production.iat[index] += highpoints[1]

        for i in range(lowpoints[0]):
            index = np.random.randint(n)
            data.production.iat[index] -= lowpoints[1]

        for i in range(drops[0]):
            start = np.random.randint(n-drops[1])
            data.production.iloc[start:start+drops[1]] = 0
            
        data.production += np.random.random(size = n)*noise
        self.data = data

if __name__ == '__main__':
    data = Decline_Generator(drops = (10,15), noise = 20, highpoints = (40,20), lowpoints = (40,20)).data

    X = data.copy()
    X['production'] = np.log(X['production'])

    # '''Add Delta Columns for differences''' 
    # for i in range(-4,5):
    #     X['delta{}'.format(i)] = X.production.diff(i)

    for i in [1,2,4,8,16,32]:
        rolling = X.production.rolling(window=1).mean()
        X['delta_rolling_{}'.format(i)] = X.production - rolling#.shift(i)

    days_ago = (X['date'].max()-X['date']).dt.days
    X['epsilon'] = days_ago/(X['production']-X['production'].iloc[-10:].mean())

    X = X.drop(['date'], axis = 1).reset_index()
    X = X.dropna()
    indices = X['index']
    X.drop('index',axis = 1, inplace = True)

    X = PCA(n_components = int(len(X.columns)/2), whiten = True).fit_transform(X)
    labels = KMeans(n_clusters = 5).fit(X).predict(X)
    data['labels'] = -1
    data['labels'].iloc[indices] = labels
    data = data[data['labels']>=0]

    colors = {0:'red',1:'green',2:'blue',3:'yellow',4:'orange',5:'black',6:'magenta'}
    data['labels'] = data['labels'].map(lambda c:colors[c])

    majority_class = data.groupby('labels').count().sort_values('date').index[-1]
    majority = data[data['labels']==majority_class].drop('labels', axis = 1)
    majority['log_production'] = np.log(majority['production'])
    
    model = LR().fit(majority['date'].values.reshape(-1,1), majority['log_production'])
    first_pass_predictions = model.predict(majority['date'].values.astype(float).reshape(-1,1))
    first_pass_residuals = first_pass_predictions - majority['log_production']

    mean, std = first_pass_residuals.mean(), first_pass_residuals.std()    
    filter = (first_pass_residuals > mean-2*std)&(first_pass_residuals<mean+2*std)

    filtered_majority = majority[filter]
    filtered_model = LR().fit(filtered_majority['date'].values.reshape(-1,1), filtered_majority['log_production'])
    prediction_dates = pd.date_range(start = majority.date.min(), end = majority.date.max() + pd.DateOffset(days = 100))
    predictions = filtered_model.predict(prediction_dates.values.astype(float).reshape(-1,1))
    predictions = np.exp(predictions)

    plt.plot(prediction_dates, predictions, color = 'green')
    plt.plot(data.set_index('date')['production'], alpha = .2)
    plt.scatter(data['date'], data['production'], c = data.labels)
    plt.show()