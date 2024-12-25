import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
import mlflow
import dagshub
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed ,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score



experiment_name = "Air_leak_compressor" 
dagshub.init(repo_owner='sfaxibrahim', repo_name='Mlops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/sfaxibrahim/Mlops.mlflow")

df=pd.read_csv("../data/processed/data_v1.csv")
df['timestamp']=pd.to_datetime(df["timestamp"])
# df.drop(columns=["Reservoirs","COMP","Caudal_impulses","Pressure_switch","H1"],inplace=True)
df.set_index('timestamp', inplace=True)
df.sort_index(inplace=True)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(columns=['Air_Leak']))  
df_scaled = pd.DataFrame(scaled_features, index=df.index, columns=df.columns[:-1])
df_scaled['Air_Leak'] = df['Air_Leak'].values
df_scaled.shape


def create_sequences(data, labels, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(labels[i+sequence_length])  # Supervised for fine-tuning
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled.drop(columns=['Air_Leak']).values, df_scaled['Air_Leak'].values)


def create_semi_supervised_lstm_autoencoder(input_shape, l2_lambda=0.001, dropout_rate=0.4):
    # Encoder
    inputs = Input(shape=input_shape)
    encoded = LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=l2(l2_lambda))(inputs)
    encoded = Dropout(dropout_rate)(encoded)
    encoded = LSTM(32, activation='relu', return_sequences=False, kernel_regularizer=l2(l2_lambda))(encoded)
    encoded = Dropout(dropout_rate)(encoded)
    
    # Decoder
    decoded = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(32, activation='relu', return_sequences=True, kernel_regularizer=l2(l2_lambda))(decoded)
    decoded = Dropout(0.4)(decoded)

    decoded = LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=l2(l2_lambda))(decoded)
    
    # Reconstruction output
    reconstruction = TimeDistributed(Dense(input_shape[1]), name="reconstruction")(decoded)
    
    # Classification Head
    classification = Dense(1, activation='sigmoid', name="classification")(encoded)

    # Define the model
    autoencoder = Model(inputs, outputs=[reconstruction, classification])
    autoencoder.compile(
        optimizer=Adam(learning_rate=0.0005), 
        loss={"reconstruction": "mean_squared_error", "classification": "binary_crossentropy"},
        loss_weights={"reconstruction": 0.5, "classification": 0.5}
    )
    
    return autoencoder

normal_data = X[y == 0]


with mlflow.start_run(run_name="Air_Leak_Semi_supervised") as run:

    #log_hypermaterts 
    mlflow.log_param("sequence_length", 10)
    mlflow.log_param("dropout_rate",0.4)
    mlflow.log_param("l2_lambda", 0.001)
    mlflow.log_param("learning_rate",0.0005)
    mlflow.log_param("batch_size",32)
    mlflow.log_param("epochs",5)


    autoencoder = create_semi_supervised_lstm_autoencoder(normal_data[0].shape)
    autoencoder.fit(
        normal_data, [normal_data, np.zeros(len(normal_data))],
        epochs=5, batch_size=32, validation_split=0.2
    )
    # Fine-tune with supervised data
    autoencoder.fit(
        X, [X, y],
        epochs=5, batch_size=32, validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    )


    # Predict using the classification head
    _, predictions = autoencoder.predict(X)
    predicted_anomalies = (predictions > 0.5).astype(int)

    # Evaluate metrics
    accuracy = accuracy_score(y, predicted_anomalies)
    roc_auc = roc_auc_score(y, predicted_anomalies)
    precision = precision_score(y, predicted_anomalies,average="binary")
    recall = recall_score(y, predicted_anomalies,average="binary")
    f1 = f1_score(y, predicted_anomalies,average="binary")
    

    # Log metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_auc",roc_auc)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)

    mlflow.sklearn.log_model(autoencoder, "model_artifact")
    
    

    print("Model and result successfully logged to Mlflow ")


