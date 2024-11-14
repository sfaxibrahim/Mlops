import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    df=pd.read_csv("../data/processed/data_v1.csv")
    df.drop(columns=["Reservoirs","COMP","Caudal_impulses","Pressure_switch","H1"],inplace=True)
    X=df.drop(columns=["Air_Leak"])
    y=df["Air_Leak"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return  X_train, X_test, y_train, y_test


def test_data(data):
    df=pd.read_csv(f"../data/test/{data}.csv")
    df.drop(columns=["Air_Leak","Reservoirs","COMP","Caudal_impulses","Pressure_switch","H1"],inplace=True)
    return df
    