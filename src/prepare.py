import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess(df):
    df = df.dropna()

    # encode categorical features
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to raw data (e.g. data/raw/telco.csv)")
    parser.add_argument("output_dir", type=str, help="Path to output directory (e.g. data/prepared)")
    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    df = load_data(input_file)
    df = preprocess(df)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train to {train_path}, test to {test_path}")


if __name__ == "__main__":
    main()
