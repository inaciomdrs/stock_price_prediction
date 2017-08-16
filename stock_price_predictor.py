'''
Stock price predictor

Script built upon tutorial at 

https://medium.com/towards-data-science/predicting-stock-prices-in-50-lines-of-python-c2c56a84b03d
'''

from sklearn.svm import SVR

import csv
import matplotlib.pyplot as plt
import numpy as np

def extract_date_and_price_from_row(row):
    date = int(row[0].split('-')[0])
    price = float(row[1])
    return date, price

def extract_dates_and_prices_from_stock_file(stock_file_name):
    dates = []
    prices = []
    with open(stock_file_name, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            date, price = extract_date_and_price_from_row(row)
            dates.append(date)
            prices.append(price)

    return dates, prices

def build_prediction_technique_on_data(dates,prices,prediction_technique):
    dates = np.reshape(dates, (len(dates), 1))
    prediction_technique.fit(dates, prices)
    return prediction_technique

def build_plot_structure(dates,prices):
    plt.scatter(dates,prices,color="black",label="Data")
    plt.xlabel('Dates')
    plt.ylabel('Price')
    plt.title('Support Vector Reg')
    plt.legend()

def plot_predictions_for(prediction_technique,color,label):
    def plot_predictions(dates, prices, x):
        plt.plot(dates,
                prediction_technique.predict(dates),
                color=color,
                label=label)
        return prediction_technique.predict(x)[0]

    return plot_predictions

def display_plot():
    plt.show()

def svr_lin(): return SVR(kernel='linear', C=1e3)
def svr_poly(): return SVR(kernel='poly', C=1e3, degree=2)
def svr_rbf(): return SVR(kernel='rbf',C=1e3, gamma=0.1)

if __name__ == '__main__':
    dataset = 'bbas3.csv'
    stock_data_dates, stock_data_prices = extract_dates_and_prices_from_stock_file(dataset)

    s_lin = build_prediction_technique_on_data(stock_data_dates,stock_data_prices,svr_lin())
    s_poly = build_prediction_technique_on_data(stock_data_dates,stock_data_prices,svr_poly())
    s_rbf = build_prediction_technique_on_data(stock_data_dates,stock_data_prices,svr_rbf())

    build_plot_structure(stock_data_dates,stock_data_prices)

    plotters = []

    plotters.append(plot_predictions_for(s_lin,"green","Linear Model"))
    plotters.append(plot_predictions_for(s_poly,"blue","Polynomial Model"))
    plotters.append(lot_predictions_for(s_rbf,"red","RBF Model"))

    for plotter in plotters:
        print(plotter(stock_data_dates,stock_data_prices,29))

    display_plot()
