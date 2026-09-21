import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_processing import (
    load_data,
    create_monthly_expenses
)


# Load data
df = load_data()

# Create monthly expense dataset
monthly = create_monthly_expenses(df)

print("\nMonthly dataset:")
print(monthly)

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

monthly = monthly.dropna()

print("\nDataset ready for training:")
print(monthly)


features = [
    "Month_Number",
    "Previous_Month",
    "Previous_2_Month",
    "Previous_3_Month"
]

X = monthly[features]

y = monthly["Total_Expense"]


model = LinearRegression()

split_index = int(
    len(monthly) * 0.8
)

train = monthly.iloc[:split_index]

test = monthly.iloc[split_index:]

X_train = train[features]

y_train = train["Total_Expense"]

X_test = test[features]

y_test = test["Total_Expense"]


model.fit(
    X_train,
    y_train
)
predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

print("\n========== MODEL PERFORMANCE ==========")

print(
    f"Mean Absolute Error: ৳{mae:,.2f}"
)

print(
    f"Root Mean Squared Error: ৳{rmse:,.2f}"
)

import matplotlib.pyplot as plt


plt.figure(figsize=(12, 5))

plt.plot(
    y_test.values,
    label="Actual"
)

plt.plot(
    predictions,
    label="Predicted"
)

plt.title(
    "Actual vs Predicted Monthly Expenses"
)

plt.xlabel("Test Month")

plt.ylabel("Expense (BDT)")

plt.legend()

plt.tight_layout()

plt.show()

joblib.dump(
    model,
    "models/expense_model.pkl"
)

print("\nModel saved successfully!")

latest = monthly.iloc[-1]

next_month_data = pd.DataFrame({
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
next_month_prediction = model.predict(
    next_month_data
)[0]

print(
    f"\nPredicted next month expense: "
    f"৳{next_month_prediction:,.2f}"
)

print("\n==================================")
print("DETAILED PREDICTIONS:")
print("==================================")

for i in range(len(test)):
    row = test.iloc[i]
    actual = row["Total_Expense"]
    predicted = predictions[i]

    print(
        f"{row['Month']}: "
        f"Actual = ৳{actual:,.2f}, "
        f"Predicted = ৳{predicted:,.2f}"
    )

print("\n==================================")