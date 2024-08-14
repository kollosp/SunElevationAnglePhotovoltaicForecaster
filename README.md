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
>>>
>>> data, ts = utils.load_dataset(convert_index_to_time=True)
>>> production = data["Production"].to_numpy()
>>>
>>> model = Model(latitude_degrees=LATITUDE_DEGREES, longitude_degrees=LONGITUDE_DEGREES, x_bins=30,
                 y_bins=60, bandwidth=0.4)
>>> model.fit(ts=ts, data=production,  zeros_filter_modifier = -0.4, density_filter_modifier = -0.5)
>>> pred = model.predict(ts) # predict production (in-sample prediction only for example)
>>> model.plot() # show model structure and its represenation
>>> plotter = Plotter(ts, [production, pred], debug=True) # run inveractive chart
>>> plotter.show()
>>> plt.show()

```
