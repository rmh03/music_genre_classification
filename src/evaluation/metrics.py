import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_sklearn_model(model, X_test, y_test, model_name, logger):
    """
    Evaluate a traditional ML model and log metrics.

    Parameters:
        model: Trained sklearn model.
        X_test: Test feature set.
        y_test: True labels.
        model_name: Name of the model.
        logger: Logger instance.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.log(f"\n=== {model_name} Evaluation ===")
    logger.log(f"Accuracy: {accuracy:.4f}")
    logger.log("Classification Report:")
    logger.log(classification_report(y_test, y_pred))
    logger.log("Confusion Matrix:")
    logger.log(str(confusion_matrix(y_test, y_pred)))
    return accuracy

def evaluate_dl_model(model, X_test, y_test, model_name, logger):
    """
    Evaluate a deep learning model and log metrics.

    Parameters:
        model: Trained deep learning model.
        X_test: Test feature set.
        y_test: True labels.
        model_name: Name of the model.
        logger: Logger instance.
    """
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    logger.log(f"\n=== {model_name} Evaluation ===")
    logger.log(f"Accuracy: {accuracy:.4f}")
    logger.log("Classification Report:")
    logger.log(classification_report(y_test, y_pred))
    logger.log("Confusion Matrix:")
    logger.log(str(confusion_matrix(y_test, y_pred)))
    return accuracy