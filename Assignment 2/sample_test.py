import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
tf.random.set_seed(42)

def evaluate():
    # Input the csv file
    """
    Sample evaluation function
    Don't modify this function
    """
    df = pd.read_csv('sample_input.csv')
     
    actual_close = np.loadtxt('sample_close.txt')
    
    pred_close = predict_func(df)
    
    # Calculation of squared_error
    actual_close = np.array(actual_close)
    pred_close = np.array(pred_close)
    mean_square_error = actual_close-pred_close


    pred_prev = [df['Close'].iloc[-1]]
    pred_prev.append(pred_close[0])
    pred_curr = pred_close
    
    actual_prev = [df['Close'].iloc[-1]]
    actual_prev.append(actual_close[0])
    actual_curr = actual_close

    # Calculation of directional_accuracy
    pred_dir = np.array(pred_curr)-np.array(pred_prev)
    actual_dir = np.array(actual_curr)-np.array(actual_prev)
    dir_accuracy = np.mean((pred_dir*actual_dir)>0)*100

    print(f'Mean Square Error: {mean_square_error: .1f}\nDirectional Accuracy: {dir_accuracy:.1f}')
    

def predict_func(data):
    """
    Modify this function to predict closing prices for next 2 samples.
    Take care of null values in the sample_input.csv file which are listed as NAN in the dataframe passed to you 
    Args:
        data (pandas Dataframe): contains the 50 continuous time series values for a stock index

    Returns:
        list (2 values): your prediction for closing price of next 2 samples
    """

    # Load the previously trained model
    model = load_model("trained_model.h5")
   
    # Handling missing values by forward fill
    df_filled = data.fillna(method='ffill')

    # Print stats of data
    data_stats = df_filled.describe()
    print(data_stats)

    # Preprocess the new sample data
    data_new = df_filled[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Close']].values

    # Scale the new data
    scaler = MinMaxScaler()
    scaler.fit(data_new)
    scaled_data = scaler.transform(data_new)

    # Create the sequences from the data
    seq_length = model.input_shape[1]
    last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, -1)

    # Prediction for next 2 days
    predictions = []
    for _ in range(2):
        next_pred = model.predict(last_sequence)
        predictions.append(next_pred[0])
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0][-1] = next_pred

    # Turn predicted values back to original scale
    predicted_prices = np.array(predictions)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    # Store the closing price in a new list
    res = [predicted_prices[0][3], predicted_prices[1][3]]
    print(res)
    return res

    # return [7214.200195, 7548.899902]
    

if __name__== "__main__":
    evaluate()