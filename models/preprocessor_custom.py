import pandas as pd

def preprocess_input(input_data, preprocessor, top_features):
    df = pd.DataFrame([input_data])
    df = df[top_features]
    X_processed = preprocessor.transform(df)
    return X_processed
