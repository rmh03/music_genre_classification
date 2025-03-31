from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, BatchNormalization, Input
import joblib

# KNN Model
def create_knn_model(n_neighbors=5, metric='minkowski', weights='uniform'):
    """
    Create a K-Nearest Neighbors (KNN) model.
    """
    return KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)

# SVM Model
def create_svm_model(kernel='rbf', C=1.0, gamma='scale'):
    """
    Create a Support Vector Machine (SVM) model.
    """
    return SVC(kernel=kernel, C=C, gamma=gamma, probability=True)

# Logistic Regression Model
def create_logreg_model(max_iter=1000, C=1.0, solver='lbfgs'):
    """
    Create a Logistic Regression model.
    """
    return LogisticRegression(max_iter=max_iter, C=C, solver=solver)

# Decision Tree Classifier
def create_decision_tree_model(max_depth=None, criterion='gini', random_state=42):
    """
    Create a Decision Tree Classifier.
    """
    return DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)

# Random Forest Classifier
def create_random_forest_model(n_estimators=100, max_depth=None, random_state=42):
    """
    Create a Random Forest Classifier.
    """
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

# CNN Model
def create_cnn_model(input_shape, num_classes):
    """
    Create a Convolutional Neural Network (CNN) model.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# LSTM Model
def create_lstm_model(input_shape, num_classes):
    """
    Create a Long Short-Term Memory (LSTM) model.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Save Model
def save_model(model, model_name, models_dir):
    """
    Save a trained model to the models directory.

    Parameters:
        model: Trained model object.
        model_name (str): Name of the model.
        models_dir (str): Directory to save the model.
    """
    model_path = f"{models_dir}/{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Load Model
def load_model(model_name, models_dir):
    """
    Load a saved model from the models directory.

    Parameters:
        model_name (str): Name of the model.
        models_dir (str): Directory where the model is saved.

    Returns:
        model: Loaded model object.
    """
    model_path = f"{models_dir}/{model_name}.pkl"
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model