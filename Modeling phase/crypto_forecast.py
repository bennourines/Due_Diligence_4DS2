import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("Starting the script...")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the cryptocurrencies to process
TARGET_CRYPTOS = [
    'Bitcoin_BTC',
    'XRP_XRP',
    'Algorand_ALGO',
    'Ethereum_ETH',
    'Solana_SOL'
]

class CryptoForecaster:
    def __init__(self, sequence_length=60, prediction_days=30):
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model()
        print("Forecaster initialized")
        
    def _build_model(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def prepare_data(self, data):
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length - self.prediction_days):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length + self.prediction_days])
            
        return np.array(X), np.array(y)
    
    def train(self, X, y, epochs=20, batch_size=64):
        print(f"Training on {len(X)} sequences...")
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    
    def predict(self, data):
        # Scale the data
        scaled_data = self.scaler.transform(data)
        
        # Create the last sequence
        last_sequence = scaled_data[-self.sequence_length:]
        last_sequence = last_sequence.reshape(1, self.sequence_length, 1)
        
        # Make prediction
        prediction = self.model.predict(last_sequence)
        prediction = self.scaler.inverse_transform(prediction)
        
        return prediction[0][0]

def process_crypto_file(file_path):
    print(f"\nProcessing file: {file_path}")
    # Read the CSV file
    df = pd.read_csv(file_path)
    print(f"Read CSV file with {len(df)} rows")
    
    # Extract cryptocurrency name from filename
    crypto_name = Path(file_path).stem.split('_')[0]
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clean the Close price column (remove $ and convert to float)
    df['Close'] = df['Close'].str.replace('$', '').str.replace(',', '').astype(float)
    print(f"Processed Close prices: min={df['Close'].min():.2f}, max={df['Close'].max():.2f}")
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Use closing price for prediction
    data = df['Close'].values.reshape(-1, 1)
    
    return crypto_name, data, df

def main():
    print("\nStarting main function...")
    # Create output directory for visualizations
    os.makedirs('crypto_predictions', exist_ok=True)
    print("Created output directory")
    
    # Initialize forecaster
    forecaster = CryptoForecaster()
    
    # Process each cryptocurrency file
    historical_data_dir = 'BnAmmar/Historical Data'
    print(f"Looking for files in: {historical_data_dir}")
    
    try:
        files = os.listdir(historical_data_dir)
        print(f"Found {len(files)} files")
    except Exception as e:
        print(f"Error listing directory: {str(e)}")
        return
    
    # Filter for target cryptocurrencies
    target_files = [f for f in files if any(crypto in f for crypto in TARGET_CRYPTOS)]
    print(f"Found {len(target_files)} target cryptocurrency files")
    
    for file in target_files:
        file_path = os.path.join(historical_data_dir, file)
        try:
            crypto_name, data, df = process_crypto_file(file_path)
            
            # Skip if not enough data
            if len(data) < forecaster.sequence_length + forecaster.prediction_days:
                print(f"Skipping {crypto_name}: Not enough data points")
                continue
            
            print(f"Processing {crypto_name}...")
            
            # Prepare data
            X, y = forecaster.prepare_data(data)
            print(f"Prepared {len(X)} sequences for training")
            
            # Split into train and test sets
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
            
            # Train the model
            forecaster.train(X_train, y_train)
            
            # Make predictions
            predictions = []
            for i in range(len(X_test)):
                pred = forecaster.predict(data[:train_size + i])
                predictions.append(pred)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            
            print(f"Metrics for {crypto_name}:")
            print(f"MSE: {mse:.2f}")
            print(f"MAE: {mae:.2f}")
            
            # Plot results
            plt.figure(figsize=(15, 6))
            plt.plot(df['Date'][-len(y_test):], y_test, label='Actual')
            plt.plot(df['Date'][-len(predictions):], predictions, label='Predicted')
            plt.title(f'{crypto_name} Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(f'crypto_predictions/{crypto_name}_prediction.png')
            plt.close()
            print(f"Saved prediction plot for {crypto_name}")
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 