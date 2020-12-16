from splib.diabetes.madex import mean_adjusted_exponent_error, graph_vs_mse
from splib.diabetes.cega import clarke_error_grid
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


if __name__ == "__main__":
    a = mean_adjusted_exponent_error([50], [40], verbose=True)

    # prepare the figures that compare madex to mse for specific reference values
    for i in [50, 120, 300]:
        plt = graph_vs_mse(i, 40, action='save', save_folder='outcome')

    # scenarios definition
    y_true = [50, 60, 100, 135, 150, 200, 250, 300]
    y_pred1 = [25, 45, 150, 160, 200, 300, 350, 390]
    y_pred2 = [75, 75, 50, 110, 100, 100, 150, 210]

    # scenarios dataframe
    df = pd.DataFrame(list(zip(y_true, y_pred1, y_pred2)), columns=[
                      'Reference Values', 'Scenario 1 predictions', 'Scenario 2 predictions'])
    print(df)
    df.to_excel('outcome/scenarios.xlsx')

    results = {}
    results['Scenario 1'] = {}
    results['Scenario 2'] = {}
    plt, zones = clarke_error_grid(y_true, y_pred1, "Scenario 1")
    results['Scenario 1']['A'] = zones[0]
    results['Scenario 1']['B'] = zones[1]
    results['Scenario 1']['C'] = zones[2]
    results['Scenario 1']['D'] = zones[3]
    results['Scenario 1']['E'] = zones[4]
    mse = mean_squared_error(y_true, y_pred1)
    results['Scenario 1']['MSE'] = round(mse, 2)
    results['Scenario 1']['RMSE'] = round(np.sqrt(mse), 2)
    madex = mean_adjusted_exponent_error(y_true, y_pred1)
    results['Scenario 1']['MADEX'] = round(madex, 2)
    results['Scenario 1']['RMADEX'] = round(np.sqrt(madex), 2)
    plt.savefig('outcome/scenario1_cega.png')

    plt, zones = clarke_error_grid(y_true, y_pred2, "Scenario 2")
    results['Scenario 2']['A'] = zones[0]
    results['Scenario 2']['B'] = zones[1]
    results['Scenario 2']['C'] = zones[2]
    results['Scenario 2']['D'] = zones[3]
    results['Scenario 2']['E'] = zones[4]
    mse = mean_squared_error(y_true, y_pred2)
    results['Scenario 2']['MSE'] = round(mse, 2)
    results['Scenario 2']['RMSE'] = round(np.sqrt(mse), 2)
    madex = mean_adjusted_exponent_error(y_true, y_pred2)
    results['Scenario 2']['MADEX'] = round(madex, 2)
    results['Scenario 2']['RMADEX'] = round(np.sqrt(madex), 2)
    plt.savefig('outcome/scenario2_cega.png')

    import pprint
    pprint.pprint(results)

    results_df = pd.DataFrame.from_dict(results, orient='index')
    print(results_df)
    results_df.to_excel('outcome/scenario_results.xlsx')

    results_df[['A', 'B', 'C', 'D', 'E']].plot(kind='bar')
    plt.xticks(rotation=0)
    plt.savefig('outcome/scenarios_zones.png')

    results_df[['MSE', 'MADEX']].plot(kind='barh')
    plt.xticks(rotation=0)
    plt.yticks(rotation=90)
    plt.savefig('outcome/compare_mse_madex.png')

    results_df[['RMSE', 'RMADEX']].plot(kind='barh')
    plt.xticks(rotation=0)
    plt.yticks(rotation=90)
    plt.savefig('outcome/compare_rmse_rmadex.png')
