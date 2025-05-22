from eda_and_preprocess import load_and_eda, preprocess

if __name__ == "__main__":
    df = load_and_eda('data/simulated_warehouse.csv')
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess(df)
