from misc import load_data
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np


def main():
    df = load_data()
    X = df.drop("MEDV", axis=1).values
    y = df["MEDV"].values

    # Baseline
    dummy = DummyRegressor(strategy="mean")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    baseline_rmse = np.mean(
        np.sqrt(-cross_val_score(dummy, X, y, cv=cv, scoring="neg_mean_squared_error"))
    )
    print(f"Baseline (mean) CV RMSE: {baseline_rmse:.3f}")

    # Final train/test using a chosen depth (e.g., 4)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = DecisionTreeRegressor(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Decision Tree Regressor Test MSE: {mse:.4f}")


if __name__ == "__main__":
    main()
