import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Data preparation
df_sat = pd.read_csv("STIN-data-set/SAT20.csv")
df_ter = pd.read_csv("STIN-data-set/TER20.csv")

frames = [df_sat, df_ter]
df_concat = pd.concat(frames)
df_concat.drop(columns='id', inplace=True)

print(df_concat.head())
print(df_concat.shape)

# sample the dataset for easier debugging
# df_concat = df_concat.sample(n=100)

# These columns give problems with NaNs after normalizing
df_concat = df_concat.drop(columns=['syn_cnt', 'urg_cnt', 'bw_psh_flag', 'fw_urg_flag', 'bw_urg_flag', 'fin_cnt', 'psh_cnt', 'ece_cnt', 'fw_byt_blk_avg', 'fw_pkt_blk_avg', 'fw_blk_rate_avg', 'bw_byt_blk_avg', 'bw_pkt_blk_avg', 'bw_blk_rate_avg'])
print(df_concat.head())

# Minority removal (merge data)
labels = {
    'Syn_DDoS': 'Syn_DDoS',
    'UDP_DDoS': 'UDP_DDoS',
    'Botnet': 'Botnet',
    'Portmap_DDoS': 'DDoS',
    'Backdoor': 'Botnet',
    'Web Attack': 'Botnet',
    'LDAP_DDoS': 'DDoS',
    'MSSQL_DDoS': 'DDoS',
    'NetBIOS_DDoS': 'DDoS'
}
df_concat[' Label'] = df_concat[' Label'].map(labels)
print(df_concat[' Label'].value_counts())

# Extract the label column
y = df_concat[' Label']
df_concat.drop(columns=' Label', inplace=True)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Print the mapping of original labels to encoded integers
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label mapping:", label_mapping)

# min max normalize
normalized_df=(df_concat-df_concat.min())/(df_concat.max()-df_concat.min())
print(normalized_df.head())

X = normalized_df

sfs = SFS(RandomForestClassifier(max_depth=2, random_state=0),
          k_features=10,
          forward=True,
          floating=False,
          scoring = 'accuracy',
          cv = 0)
sfs.fit(X,y_encoded)
print(sfs.k_feature_names_)

X = X[np.array(sfs.k_feature_names_)]
# Data prep complete, X and y_encoded are the variables to train the models with

columns_before_scaling = X.columns
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
    
# Split the data into training and testing sets
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Select the features from SFS
selected_features = list(sfs.k_feature_names_)

# Ensure that the selected_features list only contains columns that are present in the normalized_df DataFrame
selected_features = [feature for feature in selected_features if feature in normalized_df.columns]

# Get the indices of the selected features from the original DataFrame
selected_indices = [list(columns_before_scaling).index(feature) for feature in selected_features]

# Use the indices to subset the numpy arrays
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

# Reshape data for LSTM
X_train_selected = X_train_selected.reshape(X_train_selected.shape[0], len(selected_indices), 1)
X_test_selected = X_test_selected.reshape(X_test_selected.shape[0], len(selected_indices), 1)

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_selected.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=len(label_mapping), activation='softmax'))  # Softmax for multi-class classification

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_selected, y_train, epochs=10, batch_size=32, validation_data=(X_test_selected, y_test))

    
# Evaluate the model
loss, accuracy = model.evaluate(X_test_selected, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")