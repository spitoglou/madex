import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def mean_adjusted_exponent_error(y_true, y_pred, center=125, critical_range=55, slope=100, verbose=False):
    def exponent(y_hat: float, y_i: float, a=center, b=critical_range, c=slope) -> float:
        return 2 - np.tanh(((y_i - a) / b)) * ((y_hat - y_i) / c)
    sum_ = 0
    for i in range(len(y_true)):
        exp = exponent(y_pred[i], y_true[i])
        if verbose:
            print(exp)
        sum_ += abs((y_pred[i] - y_true[i])) ** exp
    return sum_ / len(y_true)


def graph_vs_mse(value, value_range, action=None, save_folder='.'):
    prediction = np.arange(value - value_range, value + value_range)
    errors = []
    mse = []
    for pred in prediction:
        errors.append(mean_adjusted_exponent_error([value], [pred]))
        mse.append(mean_squared_error([value], [pred]))
    plt.plot(prediction, errors, label='madex')
    plt.plot(prediction, mse, label='mse', ls='dotted')
    plt.axvline(value, label='Reference Value', color='k', ls='--')
    plt.xlabel('Predicted Value')
    plt.ylabel('Error')
    plt.title('{} +- {}'.format(value, value_range))
    plt.legend()
    if action == 'save':
        plt.savefig(
            f'{save_folder}/compare_vs_mse({value}+-{value_range}).png')
        plt.clf()
    else:
        return plt
