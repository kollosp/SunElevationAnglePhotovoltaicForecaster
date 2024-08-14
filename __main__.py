from datasets import utils
from SEAPF.Model import Model
from matplotlib import pyplot as plt
from SEAPF.Plotter import Plotter
from sklearn.metrics import mean_absolute_error

SAMPLES_PER_HOUR=12
SAMPLES_PER_DAY=12

"""
    Show dataset if the script was run directly instead of being loaded as package
"""
if __name__ == "__main__":
    data, ts = utils.load_dataset(convert_index_to_time=True)
    # print(data.head(), data.columns)

    # Use all columns to create X
    # X = utils.timeseries_to_dataset([data[i] for i in data.columns], window_size=1)

    y = data["Production"].to_numpy()
    X =  ts.reshape(1,-1).T #make 2D


    model = Model(window_size=SAMPLES_PER_HOUR*3, latitude_degrees=utils.LATITUDE_DEGREES, longitude_degrees=utils.LONGITUDE_DEGREES, x_bins=30,
                 y_bins=60, bandwidth=0.4, interpolation=True)
    # in sample training - only for test
    model.fit(X=X, y=y,  zeros_filter_modifier = -0.2, density_filter_modifier = -0.4)
    pred = model.predict(X)

    model.plot()  # for displaying model parameters
    plotter = Plotter(ts, [y, pred], debug=True)

    print(f"In-sample MAE: {mean_absolute_error(y, pred)}")

    plotter.show()
    plt.show()
