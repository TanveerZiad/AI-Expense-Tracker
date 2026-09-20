import pandas as pd


def load_data(filepath="data/expenses.csv"):
    df = pd.read_csv(filepath)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


if __name__ == "__main__":
    df = load_data()

    print(df.head())
    print("\nDataset shape:")
    print(df.shape)

    print("\nDataset information:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())