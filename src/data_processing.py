import pandas as pd


FILE_PATH = "data/expenses_income_summary.csv"


def load_data():

    df = pd.read_csv(FILE_PATH)

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Amount_BDT"] = pd.to_numeric(
        df["Amount_BDT"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date", "Amount_BDT"]
    )

    return df


def get_expenses(df):

    return df[
        df["Transaction_Type"].str.lower() == "expense"
    ].copy()


def get_income(df):

    return df[
        df["Transaction_Type"].str.lower() == "income"
    ].copy()


def create_monthly_expenses(df):

    expenses = get_expenses(df)

    expenses["Month"] = (
        expenses["Date"]
        .dt.to_period("M")
    )

    monthly = (
        expenses
        .groupby("Month")["Amount_BDT"]
        .sum()
        .reset_index()
    )

    monthly.columns = [
        "Month",
        "Total_Expense"
    ]

    return monthly


if __name__ == "__main__":

    df = load_data()

    expenses = get_expenses(df)

    income = get_income(df)

    print("\n========== SUMMARY ==========")

    print("Total transactions:", len(df))

    print("Total expenses:", len(expenses))

    print("Total income:", len(income))

    print(
        "Total expense amount:",
        expenses["Amount_BDT"].sum()
    )

    print(
        "Total income amount:",
        income["Amount_BDT"].sum()
    )

    print("\n========== MONTHLY EXPENSES ==========")

    monthly = create_monthly_expenses(df)

    print(monthly)