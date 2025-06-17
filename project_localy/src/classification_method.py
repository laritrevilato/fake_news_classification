from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def svc_classifier(x_train, y_train, x_test, y_test):
    """Train and evaluate SVC with linear kernel."""
    model = SVC(kernel="linear", C=1.0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        # "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def logistic_regression_classifier(x_train, y_train, x_test, y_test):
    """Train and evaluate Logistic Regression."""
    model = LogisticRegression(max_iter=500)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        # "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def multinomial_nb_classifier(x_train, y_train, x_test, y_test):
    """Train and evaluate Multinomial Naive Bayes."""
    model = MultinomialNB(alpha=1.0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        # "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def random_forest_classifier(x_train, y_train, x_test, y_test):
    """Train and evaluate Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=42, n_jobs=-1
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        # "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
