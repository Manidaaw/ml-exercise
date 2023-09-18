import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["id"].values
    predictions = None

    for FOLD in range(5):
        df = pd.read_csv(TEST_DATA)
        encoders = joblib.load(f"models/{MODEL}_{FOLD}_label_encoder.pkl")
        cols = joblib.load(f"models/{MODEL}_{FOLD}_columns.pkl")
        for col in cols:
            label = encoders[col]
            df.loc[:, col] = label.transform(df[col].values.tolist())

        #data is ready to train
        clf = joblib.load(f"models/{MODEL}_{FOLD}.pkl")

        df = df[cols]
        preds = clf.predict_proba(df)[:,1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx.astype(int), predictions)), columns=["id", "target"])
    return sub

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"models/{MODEL}.csv", index=False)