# SunElevationAnglePhotovoltaicForecaster



This repository stores `SEAPF` model which is provided under the directory `SEAPF`

Project structure:
```
.
├── datasets
│   ├── database.csv
│   ├── Dataset.py
│   ├── __init__.py
│   └── utils.py
├── experimental
│   ├── Experimental.py
│   └── __init__.py
├── __main__.py
├── README.md
├── SEAPF
│   ├── __init__.py
│   ├── Model.py
│   ├── Optimized.py
│   ├── Overlay.py
│   ├── Plotter.py
│   ├── README.md
│   ├── setup.py
│   ├── tests
│   └── Time.py
├── Solar
│   ├── __init__.py
│   ├── README.md
│   └── Solar.py
├── Transition_Into_Solar_Elevation_Angle_Domain_for_Photovoltaic_Power_Generation_Forecasting.pdf
└── Transition_Into_Solar_Elevation_Angle_Domain_for_Photovoltaic_Power_Generation_Forecasting___POSTER.pdf

```


## Example
To use the model the following snipped may be used
```python

>>> from SEAPF.Model import Model
>>> from datasets import utils
>>> from matplotlib import pyplot as plt
>>> from SEAPF.Plotter import Plotter
>>> 
>>> ts = ... # load unix timestamps in some way
>>> data = ... # load data in some way
>>> # Optional: load from databases directory
>>> data, ts = utils.load_dataset(convert_index_to_time=True)
>>> production = data["Production"].to_numpy()
>>> y = data["Production"].to_numpy()
>>> print("y:\n", y)
y:
 [0. 0. 0. ... 0. 0. 0.]

>>> X =  ts.reshape(1,-1).T #make 2D
>>> print("X:\n",X)
X:
 [[1.6565364e+09]
 [1.6565367e+09]
 [1.6565370e+09]
 ...
 [1.6732179e+09]
 [1.6732182e+09]
 [1.6732185e+09]]

>>>
>>> model = Model(window_size=SAMPLES_PER_HOUR*3,
>>>              latitude_degrees=utils.LATITUDE_DEGREES,
>>>              longitude_degrees=utils.LONGITUDE_DEGREES,
>>>              x_bins=30,
>>>              y_bins=60,
>>>              bandwidth=0.4,
>>>              interpolation=True)
>>> # in sample prediction - only for test!
>>> model.fit(X=X, y=y,  zeros_filter_modifier = -0.2, density_filter_modifier = -0.4)>>> pred = model.predict(ts) # predict production (in-sample prediction only for example)
>>> model.plot() # show model structure and its represenation
>>> model.plot()  # for displaying model parameters
>>> plotter = Plotter(ts, [production, pred], debug=True) # run inveractive chart
>>> plotter.show()
>>> plt.show()

```
