import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def add_data(hmap, data, ts):
    if ts not in hmap:
        hmap[ts] = data
    elif data != hmap[ts]:
        print(f"WHY {ts} is already in: {hmap}")

training_files = ["../data/bicep_curl/suzan_bicep_set1.log", "../data/bicep_curl/jake_bicep_set1.log", "../data/bicep_curl/udai_bicep_set1.log", "../data/shoulder_press/suzan_shoulder_set1.log", "../data/shoulder_press/jake_shoulder_set1.log", "../data/shoulder_press/udai_shoulder_set1.log", "../data/row/suzan_row_set1.log", "../data/row/jake_row_set1.log", "../data/row/udai_row_set1.log", "../data/rdl/suzan_rdl_set1.log", "../data/rdl/jake_rdl_set1.log", "../data/rdl/jessica_rdl_set1.log", "../data/squat/suzan_squat_set1.log", "../data/squat/jake_squat_set1.log", "../data/squat/udai_squat_set1.log"]
test_files = ["../data/bicep_curl/suzan_bicep_set2.log", "../data/bicep_curl/jake_bicep_set2.log", "../data/bicep_curl/udai_bicep_set2.log", "../data/shoulder_press/suzan_shoulder_set2.log", "../data/shoulder_press/jake_shoulder_set2.log", "../data/shoulder_press/udai_shoulder_set2.log", "../data/row/suzan_row_set2.log", "../data/row/jake_row_set2.log", "../data/row/udai_row_set2.log", "../data/rdl/suzan_rdl_set2.log", "../data/rdl/jake_rdl_set2.log", "../data/rdl/jessica_rdl_set2.log", "../data/squat/suzan_squat_set2.log", "../data/squat/jake_squat_set2.log", "../data/squat/udai_squat_set2.log"]
labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
class_names = ["bicep_curl", "shoulder_press", "row", "rdl", "squat"]
# 0 = bicep curl, 1 = shoulder press, 2 = row, 3 = rdl, 4 = squat
# labels = ["bicep_curl", "bicep_curl", "bicep_curl", "shoulder_press", "shoulder_press", "shoulder_press", "row", "row", "row", "rdl", "rdl", "rdl", "squat", "squat", "squat"]

def create_df(filename):
    print(filename)
    with open(filename, 'r') as file:
        timestamps = []
        accel_x = {}
        accel_y = {}
        accel_z = {}
        gyro_x = {}
        gyro_y = {}
        gyro_z = {}

        for line in file:
            data_line = line.strip()
            if data_line and data_line[0].isdigit():
                # print(data_line)
                _, data_type, timestamp, data = data_line.split(",")
                timestamp = int(timestamp)
                data = float(data)
                idx = None
                if timestamp not in timestamps:
                    timestamps.append(timestamp)
                if (data_type == "AX"):
                    add_data(accel_x, data, timestamp)
                elif (data_type == "AY"):
                    add_data(accel_y, data, timestamp)
                elif (data_type == "AZ"):
                    add_data(accel_z, data, timestamp)
                elif (data_type == "GX"):
                    add_data(gyro_x, data, timestamp)
                elif (data_type == "GY"):
                    add_data(gyro_y, data, timestamp)
                elif (data_type == "GZ"):
                    add_data(gyro_z, data, timestamp)
        
        for ts in timestamps:
            if ts not in accel_x:
                accel_x[ts] = None
            if ts not in accel_y:
                accel_y[ts] = None
            if ts not in accel_z:
                accel_z[ts] = None
            if ts not in gyro_x:
                gyro_x[ts] = None
            if ts not in gyro_y:
                gyro_y[ts] = None
            if ts not in gyro_z:
                gyro_z[ts] = None


        accel_x = dict(sorted(accel_x.items()))
        accel_y = dict(sorted(accel_y.items()))
        accel_z = dict(sorted(accel_z.items()))
        gyro_x = dict(sorted(gyro_x.items()))
        gyro_y = dict(sorted(gyro_y.items()))
        gyro_z = dict(sorted(gyro_z.items()))
                    
        # print(f"timestamps: {timestamps}")
        # print(len(timestamps))
        # print(f"accel_x: {accel_x}")
        # print(len(accel_x))
        # print(f"accel_y: {accel_y}")
        # print(len(accel_y))
        # print(f"accel_z: {accel_z}")
        # print(len(accel_z))
        # print(f"gyro_x: {gyro_x}")
        # print(len(gyro_x))
        # print(f"gyro_y: {gyro_y}")
        # print(len(gyro_y))
        # print(f"gyro_z: {gyro_z}")
        # print(len(gyro_z))

        df = pd.DataFrame({'timestamp': timestamps, 'accel_x': accel_x.values(), 'accel_y': accel_y.values(), 'accel_z': accel_z.values(), 'gyro_x': gyro_x.values(), 'gyro_y': gyro_y.values(), 'gyro_z': gyro_z.values()})
        # df.to_csv('raw_df.csv')
        cleaned_df = df.interpolate(method="linear", limit_direction="both")
        name = "../csvs/" + filename.split("/")[-1].split(".")[0] + "_df.csv"
        if len(sys.argv) > 1 and sys.argv[1] == "write":
            cleaned_df.to_csv(name, index=False)
        print(cleaned_df.head())
        # dfs.append(cleaned_df)
        return cleaned_df
        # cleaned_df.plot(x="timestamp")
        # plt.title("Bicep Curls")
        # plt.xlabel("Timestamp")
        # plt.ylabel("IMU Data")
        # plt.show()


X_train = []
X_test = []

for f in training_files:
    X_train.append(create_df(f))

for f in test_files:
    X_test.append(create_df(f))

y_train = np.array(labels)
y_test = np.array(labels)

max_len = 100

def pad(df, max_len):
    arr = df[['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z']].values
    if len(arr) >= max_len:
        return arr[:max_len]
    else:
        pad = np.zeros((max_len - len(arr), 6))
        return np.vstack((arr, pad))


X_train = np.array([pad(df, max_len) for df in X_train])
X_test = np.array([pad(df, max_len) for df in X_test])

num_classes = 5
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dense(64, activation='sigmoid'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint_filepath = '/tmp/ckpt/checkpoint.model.keras'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    verbose=1,
    callbacks=[model_checkpoint_callback]
)

model.load_weights(checkpoint_filepath)

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)

print(f"y_pred: {y_pred}")
print(f"y_true: {y_true}")

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Classification Report:\n", classification_report(y_true, y_pred))

print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names) # class_names are optional
disp.plot()
plt.show()