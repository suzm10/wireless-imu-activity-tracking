import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

def create_df(filename, write=False):
    print(filename)
    with open(filename, 'r') as file:
        timestamps = []
        accel_x = []
        accel_y = []
        accel_z = []
        gyro_x = []
        gyro_y = []
        gyro_z = []

        for line in file:
            data_line = line.strip()
            if data_line and data_line[0].isdigit():
                # print(data_line)
                node_num, timestamp, ax, ay, az, gx, gy, gz, mx, my, mz = data_line.split(",")
                timestamps.append(int(timestamp))
                accel_x.append(float(ax))
                accel_y.append(float(ay))
                accel_z.append(float(az))
                gyro_x.append(float(gx))
                gyro_y.append(float(gy))
                gyro_z.append(float(gz))
        
        # it overfits when i normalize
        # accel_x = (accel_x - np.mean(accel_x)) / np.std(accel_x)
        # accel_y = (accel_y - np.mean(accel_y)) / np.std(accel_y)
        # accel_z = (accel_z - np.mean(accel_z)) / np.std(accel_z)
        # gyro_x = (gyro_x - np.mean(gyro_x)) / np.std(gyro_x)
        # gyro_y = (gyro_y - np.mean(gyro_y)) / np.std(gyro_y)
        # gyro_z = (gyro_z - np.mean(gyro_z)) / np.std(gyro_z)

        df = pd.DataFrame({'timestamp': timestamps, 
                           'accel_x': accel_x, 
                           'accel_y': accel_y, 
                           'accel_z': accel_z, 
                           'gyro_x': gyro_x, 
                           'gyro_y': gyro_y, 
                           'gyro_z': gyro_z})

        cleaned_df = df.interpolate(method="linear", limit_direction="both")
        name = "../csvs2/" + filename.split("/")[-1].split(".")[0] + "_df.csv"
        if write:
            cleaned_df.to_csv(name, index=False)
        #print(cleaned_df.head())
        return cleaned_df
    
def get_peaks_and_valleys(df, dist=100):
    peaks, _ = find_peaks(df["accel_z"], height=0, distance=dist)
    valleys, _ = find_peaks(-1 * df["accel_z"], height=0, distance=dist)

    peakvals = [(int(peak), float(df['accel_z'].iloc[peak])) for peak in peaks]
    valleyvals = [(int(valley), float(df['accel_z'].iloc[valley])) for valley in valleys]
    print(f"peakvals: {peakvals}")
    print(f"valleyvals: {valleyvals}")

    print(f"peaks: {peaks}, peakvals: {peakvals}")
    print(f"valleys: {valleys}, valleyvals: {valleyvals}")
    return peaks, valleys, peakvals, valleyvals

def plot_df(df, title, peaks=None, valleys=None):
    plt.figure(figsize=(10, 8))
    # plt.plot(x, y, ...)
    plt.plot(df["timestamp"], df["accel_x"], label="accel_x", marker='o')
    plt.plot(df["timestamp"], df["accel_y"], label="accel_y", marker='o')
    plt.plot(df["timestamp"], df["accel_z"], label="accel_z", marker='o')
    plt.plot(df["timestamp"], df["gyro_x"], label="gyro_x", marker='o')
    plt.plot(df["timestamp"], df["gyro_y"], label="gyro_y", marker='o')
    plt.plot(df["timestamp"], df["gyro_z"], label="gyro_z", marker='o')

    if peaks is not None and valleys is not None:
        plt.plot(df["timestamp"].iloc[peaks], df["accel_z"].iloc[peaks], "rx", label="peaks")
        plt.plot(df["timestamp"].iloc[valleys], df["accel_z"].iloc[valleys], "ro", label="valleys")

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.xlabel("Timestamp")
    plt.ylabel("IMU Data")
    plt.show()

def get_num_reps(df, peaks, valleys):
    valleyvals = [valley[1] for valley in valleys]
    valleyidxs = [valley[0] for valley in valleys]
    peakvals = [peak[1] for peak in peaks]
    peakidxs = [peak[0] for peak in peaks]

    valleymean = np.mean(valleyvals)
    print(valleymean)
    valley_outlier_indices = np.where(np.abs(valleyvals - valleymean) > 5)[0]
    print(f"valley_outlier_indices: {valley_outlier_indices}")

    peakmean = np.mean(peakvals)
    print(peakmean)
    peak_outlier_indices = np.where(np.abs(peakvals - peakmean) > 5)[0]
    print(f"peak_outlier_indices: {peak_outlier_indices}")

    valid_valley_idxs = [x for i, x in enumerate(valleyidxs) if i not in valley_outlier_indices]
    valid_peak_idxs = [x for i, x in enumerate(peakidxs) if i not in peak_outlier_indices]

    print(f"valid_valley_idxs: {valid_valley_idxs}")
    print(f"valid_peak_idxs: {valid_peak_idxs}")
    start = 0
    end = len(df)
    if valid_peak_idxs and valid_valley_idxs:
        start = max(valid_valley_idxs[0], valid_peak_idxs[0])
        end = max(valid_valley_idxs[-1], valid_peak_idxs[-1])
    print(f"start: {start}, end: {end}")


    num_valid_valleys = len(valleyidxs) - len(valley_outlier_indices)
    num_valid_peaks = len(peakidxs) - len(peak_outlier_indices)
    print(f"num_valid_peaks: {num_valid_peaks}, num_valid_valleys: {num_valid_valleys}")
    return start, end, num_valid_peaks, num_valid_valleys

    # Q1 = np.percentile(valleyvals, 90, method='midpoint')
    # print(Q1)
    # valley_outlier_indices = np.where(valleyvals > Q1)[0]
    # print(valley_outlier_indices)

    # Q1 = np.percentile(peakvals, 5, method='midpoint')
    # print(Q1)
    # peak_outlier_indices = np.where(peakvals < Q1)[0]
    # print(peak_outlier_indices)

def smooth_and_resample(df, target_len=300, window_length=11, polyorder=2):
    """
    Smooths and resamples a DataFrame of IMU data to have target_len rows.
    
    Parameters:
      df: pandas DataFrame containing accelerometer/gyroscope columns.
      target_len: desired number of samples (rows) per file.
      window_length: window size for smoothing (odd number).
      polyorder: polynomial order for Savitzky-Golay filter.
    """
    
    # Keep only numeric columns (assumes these are your sensor readings)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Smooth each numeric column with a Savitzky-Golay filter
    smoothed = df[numeric_cols].apply(
        lambda col: savgol_filter(col, window_length=min(window_length, len(col)//2*2+1), polyorder=polyorder)
    )
    
    # Resample (interpolate) to get target_len rows
    original_len = len(smoothed)
    new_idx = np.linspace(0, original_len - 1, target_len)
    
    resampled = pd.DataFrame(
        {col: np.interp(new_idx, np.arange(original_len), smoothed[col]) for col in smoothed.columns}
    )
    #print(resampled.head())
    return resampled

# Use find_peaks to segment data into individual reps
def segment_into_reps(df, min_rep_length=15, max_rep_length=200):
    from scipy.signal import find_peaks
    
    # Use find_peaks with the total magnitude of acceleration
    # Make sure to set prominence to a decent value to avoid false positives
    # NOTE: Distance between peaks may vary! Needs resampling!
    peaks, _ = find_peaks(df['accel_mag'], distance=40, prominence=5)
        
    reps = []

    # Iterate through each pair of peaks we have    
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i+1]

        # Get the data points in between the peaks
        rep_df = df.iloc[start:end].copy()
        
        # Append to result if the distance between peaks within the specified bounds
        rep_len = len(rep_df)
        if min_rep_length <= rep_len <= max_rep_length:
            reps.append(rep_df)
    
    print("REPS: ", len(reps))

    return reps

# Adding features to the dataset to provide more information to the models
# Restricting to rotation-invariant methods since we cannot confirm orientation of barbell
def add_features(df):  

    # Magnitudes
    # (should be helpful given orientation is an unknown)
    df['accel_mag'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
    df['gyro_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
        
    # Gyro-accel ratio
    # (because why not. ¯\_(ツ)_/¯)
    df['gyro_accel_ratio'] = df['gyro_mag'] / (df['accel_mag'] + 1e-8) # + 1e-8 to avoid div by 0
    
    # Jerk
    # (smoothing and resampling should help given some noisy data)
    df['jerk_y'] = df['accel_y'].diff().fillna(0)
    df['jerk_z'] = df['accel_z'].diff().fillna(0)

    # Magnitude of the jerk
    # (wait that sounds a bit funny)
    df['jerk_mag'] = np.sqrt(df['jerk_y']**2 + df['jerk_z']**2)

    # Smoothed jerk
    # (and that sounds even worse)
    df['jerk_smooth'] = df['jerk_mag'].rolling(window=5, center=True).mean().fillna(0)
    
    return df