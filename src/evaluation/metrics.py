import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_sklearn_model(model, X_test, y_test, model_name, logger):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    logger.log(f"\n=== {model_name} Evaluation ===")
    logger.log(f"Accuracy: {accuracy:.4f}")
    logger.log(f"Precision: {precision:.4f}")
    logger.log(f"Recall: {recall:.4f}")
    logger.log(f"F1 Score: {f1:.4f}")
    logger.log("Confusion Matrix:")
    logger.log(str(confusion_matrix(y_test, y_pred)))

def evaluate_dl_model(model, X_test, y_test, model_name, logger):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    logger.log(f"\n=== {model_name} Evaluation ===")
    logger.log(f"Accuracy: {accuracy:.4f}")
    logger.log(f"Precision: {precision:.4f}")
    logger.log(f"Recall: {recall:.4f}")
    logger.log(f"F1 Score: {f1:.4f}")
    logger.log("Confusion Matrix:")
    logger.log(str(confusion_matrix(y_test, y_pred)))