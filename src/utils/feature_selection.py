from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def select_features(X, y):
    """
    Select important features using a Random Forest model.

    Parameters:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target labels.

    Returns:
        numpy.ndarray: Reduced feature matrix with selected features.
    """
    # Train a Random Forest model to determine feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Use SelectFromModel to select important features
    selector = SelectFromModel(rf, prefit=True)
    X_selected = selector.transform(X)

    return X_selected