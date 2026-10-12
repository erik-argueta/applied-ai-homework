import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# load dataset

X = np.array(
    [[-3.0], [-2.5], [-2.0], [-1.5], [-1.0], [0.0], [1.0], [1.5], [2.0], [2.5], [2.7]]
)
Y = np.array([15.5, 12.9, 9.5, 6.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 21.4])

# displaying the scatter plot of the dataset

plt.scatter(X[:, 0], Y, color="blue", label="Training Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Training Dataset")
plt.legend()
plt.show()

# time to train the linear regression model
model = LinearRegression()
model.fit(X, Y)

# now to display the coefficient vector
w0 = model.intercept_
w = model.coef_

# recall that coefficient vectors are [wo, w1, ... wn]
coefficients = np.concatenate(([w0], w))
print("\nCoefficient vector [w0, w1, ..., wn]:")
print(coefficients)

equation = f"y = {w0:.4f}"
for i, coefficient in enumerate(w, start=1):
    equation += f" + ({coefficient:.4f})*x{i}"
print("\nPolynomial equation:")
print(equation)

# now i need to compute the training errors using the RMSE
y_pred = model.predict(X)
rmse = np.sqrt(np.sum((Y - y_pred) ** 2) / len(Y))
print(f"\nTraining RMSE: {rmse:.4f}")
# checkin against skleanr
print(f"Check (sklearn): {np.sqrt(mean_squared_error(Y, y_pred)):.4f}")

# load dataset from gas properties, split into 80-20
df = pd.read_csv("GasProperties.csv")

feature_cols = ["T", "P", "TC", "SV"]
target_col = "Idx"

features = df[feature_cols].to_numpy(dtype=float)
target = df[target_col].to_numpy(dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

Xb_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
Xb_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

w_hat = np.linalg.inv(Xb_train.T @ Xb_train) @ Xb_train.T @ y_train

print("\nGasProperties coefficient vector [w0, w_T, w_P, w_TC, w_SV]:")
print(w_hat)


def rmse_fn(y_true, y_hat):
    return np.sqrt(np.sum((y_true - y_hat) ** 2) / len(y_true))


print(f"Training RMSE: {rmse_fn(y_train, Xb_train @ w_hat):.4f}")
print(f"Testing RMSE:  {rmse_fn(y_test, Xb_test @ w_hat):.4f}")

# Optional sanity check: sklearn should give the same coefficients
check = LinearRegression().fit(X_train, y_train)
print("sklearn check:", np.concatenate(([check.intercept_], check.coef_)))
