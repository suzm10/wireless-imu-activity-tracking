# Activity Tracking with IMU's

Our data is currently collected with the IMU attached to the center of the barbell.

## Data Collection

We used an Adafruit Feather nRF52840 Sense for the purposes of data collection, leveraging components such as its accelerometer, gyroscope, and analog-to-digital signal reading capabilities.

While the format of the CSV output has evolved since the beginning (e.g. how the data in ``data2/`` lacks any velostat/weight readings and magnetometer readings), the final version reflected in ``velostat/`` (barring the raw analog value not being converted to estimated weight) has the following format:

```
nodeNum,timestamp,estWeight,accelX,accelY,accelZ,gyroX,gyroY,gyroZ,magX,magY,magZ
```

The microcontroller loop can be found in ``Arduino/initialTesting/accelBluetoothDemo/accelBluetoothDemo.ino``. The factory demo found in the ``Arduino`` folder is merely just example code from the manufacturer and was only used for testing purposes.

## Data Analysis

The best results were achieved with a Random Forest Classifier (via Scikit-Learn). The corresponding notebook can be found in ``data_process/features.ipynb``.

The "alternative method" with deep learning models is also provided here for context. The corresponding notebook file can be found in ``data_process/features.ipynb``.

Additional auxillary files (e.g. helper functions, visualization notebooks, etc.) for data processing and evaluation can be found in ``data_process/``.

Note that the most recent data that was used in training and evaluating these models can be found in ``data2/``.

### Bicep Curl

What the data looks like:

![alt text](bicep_curl.png)

### Confusion Matrix

87% classification accuracy:

![alt text](confusion_matrix.png)

## Weight Estimation with Velostat

In the ``velostat/`` folder, you will find the CSV files showing the baseline measurement data in files such as ``timed_velostat_5.csv``. The small python script to get the parameters that were plugged into the Arduino code can also be found in ``velostat_estimate_lbs.py``.

More info on the results from this section can be found in the slides (also provided in this zip file as a PDF file).
