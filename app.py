import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="AI Expense Tracker",
    page_icon="💰",
    layout="wide"
)


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "data/expenses_income_summary.csv"
)

df["Date"] = pd.to_datetime(df["Date"])

df["Amount_BDT"] = pd.to_numeric(
    df["Amount_BDT"],
    errors="coerce"
)


# =========================
# TITLE
# =========================

st.title("💰 AI Expense Tracker")

st.write(
    "Track spending, analyze financial patterns, "
    "and predict next month's expenses."
)


# =========================
# FILTER EXPENSES
# =========================

expenses = df[
    df["Transaction_Type"].str.lower() == "expense"
].copy()

income = df[
    df["Transaction_Type"].str.lower() == "income"
].copy()


# =========================
# SUMMARY
# =========================

total_expense = expenses["Amount_BDT"].sum()

total_income = income["Amount_BDT"].sum()

transaction_count = len(df)


col1, col2, col3 = st.columns(3)


col1.metric(
    "Total Income",
    f"৳{total_income:,.0f}"
)

col2.metric(
    "Total Expenses",
    f"৳{total_expense:,.0f}"
)

col3.metric(
    "Transactions",
    f"{transaction_count:,}"
)


# =========================
# CATEGORY ANALYSIS
# =========================

st.subheader("📊 Spending by Category")


category_expenses = (
    expenses
    .groupby("Category")["Amount_BDT"]
    .sum()
    .sort_values(ascending=False)
)


st.bar_chart(category_expenses)


# =========================
# MONTHLY EXPENSES
# =========================

st.subheader("📈 Monthly Expenses")


expenses["Month"] = (
    expenses["Date"]
    .dt.to_period("M")
)

monthly_expenses = (
    expenses
    .groupby("Month")["Amount_BDT"]
    .sum()
)

monthly_expenses.index = (
    monthly_expenses.index.astype(str)
)


st.line_chart(monthly_expenses)


# =========================
# ML PREDICTION
# =========================

st.subheader("🤖 Next Month Prediction")


model = joblib.load(
    "models/expense_model.pkl"
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


monthly["Month_Number"] = range(
    1,
    len(monthly) + 1
)

monthly["Previous_Month"] = (
    monthly["Total_Expense"].shift(1)
)

monthly["Previous_2_Month"] = (
    monthly["Total_Expense"].shift(2)
)

monthly["Previous_3_Month"] = (
    monthly["Total_Expense"].shift(3)
)


latest = monthly.iloc[-1]


prediction_data = pd.DataFrame({

    "Month_Number": [
        latest["Month_Number"] + 1
    ],

    "Previous_Month": [
        latest["Total_Expense"]
    ],

    "Previous_2_Month": [
        monthly.iloc[-2]["Total_Expense"]
    ],

    "Previous_3_Month": [
        monthly.iloc[-3]["Total_Expense"]
    ]
})


prediction = model.predict(
    prediction_data
)[0]


st.metric(
    "Predicted Expense",
    f"৳{prediction:,.0f}"
)