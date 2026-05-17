"""Generated from Jupyter notebook: notebook

Magics and shell lines are commented out. Run with a normal Python interpreter."""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def main():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression().fit(X, y)
    coefs = model.coef_
    intercepts = model.intercept_
    preds = X @ coefs.T + intercepts


def main() -> None:
    main()


if __name__ == "__main__":
    main()
