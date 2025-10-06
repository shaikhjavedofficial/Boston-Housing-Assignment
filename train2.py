from misc import load_data, preprocess_data, train_model, evaluate_model
from sklearn.kernel_ridge import KernelRidge
import numpy as np


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    var = np.var(X_train, axis=0).mean()
    n_features = X_train.shape[1]
    gamma = 1.0 / (n_features * var + 1e-8)

    model = KernelRidge(alpha=0.1, kernel="rbf", gamma=gamma)
    model = train_model(X_train, y_train, model)
    mse = evaluate_model(X_test, y_test, model)
    print(f"Kernel Ridge Regressor MSE: {mse:.4f}")


if __name__ == "__main__":
    main()
