import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
tf.random.set_seed(42)

def train_model(df):
    # Preparing the data
    data = df.drop(columns=['Date']).values

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequence and label 
    seq_length = 50
    X, y = create_sequences(scaled_data, seq_length)
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units = 64, input_shape=(seq_length, data.shape[1])))
    model.add(Dense(units = data.shape[1]))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Train the model
    model.fit(X, y, epochs = 10, batch_size = 32)

    # Save the trained model
    model.save("trained_model.h5")
    return model

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length][-1])  
    return np.array(X), np.array(y)

if __name__ == "__main__":
    df = pd.read_csv("STOCK INDEX.csv")

    # Handling missing values
    df_filled = df.fillna(method = 'ffill')

    model = train_model(df_filled)
    print(model.summary())
    
    
