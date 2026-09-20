import pandas as pd


def load_data(filepath="data/expenses.csv"):
    df = pd.read_csv(filepath)

    # Standardize column names for date and amount
    rename_mapping = {}
    if "Date" in df.columns and "date" not in df.columns:
        rename_mapping["Date"] = "date"
    if "Amount_BDT" in df.columns and "amount" not in df.columns:
        rename_mapping["Amount_BDT"] = "amount"
    elif "Amount" in df.columns and "amount" not in df.columns:
        rename_mapping["Amount"] = "amount"

    if rename_mapping:
        df = df.rename(columns=rename_mapping)

    df["date"] = pd.to_datetime(df["date"])

    return df


def create_monthly_data(df):
    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M")

    monthly = (
        df.groupby("year_month")["amount"]
        .sum()
        .reset_index()
    )

    monthly["year_month"] = monthly["year_month"].astype(str)

    return monthly


if __name__ == "__main__":

    df = load_data()

    print("Raw Data:")
    print(df.head())

    monthly = create_monthly_data(df)

    print("\nMonthly Expenses:")
    print(monthly)