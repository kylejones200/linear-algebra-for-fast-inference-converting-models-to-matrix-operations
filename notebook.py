"""Generated from Jupyter notebook: notebook

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def main():
    # --- code cell ---

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression().fit(X, y)


    # --- code cell ---

    coefs = model.coef_
    intercepts = model.intercept_
    preds = X @ coefs.T + intercepts


if __name__ == "__main__":
    main()
