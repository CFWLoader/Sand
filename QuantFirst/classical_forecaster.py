from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


def arima_automata(series):
    # f1 = plt.figure()
    # plot_pacf(series)
    # plt.plot(series.diff())
    arima_model = ARIMA(series, order=(1, 2, 1)).fit()
    # arima_model.plot_predict(dynamic=False)
    # plt.plot(arima_model.forecast())
    plt.show()