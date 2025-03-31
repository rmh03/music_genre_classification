# Audio Genre Classification Project

## Project Overview
This project focuses on classifying audio files into different genres using machine learning and deep learning techniques. The dataset consists of audio files from various genres such as blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock. The project involves feature extraction, data preprocessing, model training, evaluation, and visualization.

## Features
- **Audio Feature Extraction**:
  - Extracted features include Chroma STFT, RMS Energy, Spectral Centroid, Spectral Bandwidth, Zero Crossing Rate, and MFCCs.
- **Data Augmentation**:
  - Augmented audio files to improve model generalization.
- **Machine Learning Models**:
  - K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest, and Gradient Boosting.
- **Deep Learning Models**:
  - Convolutional Neural Networks (CNN).
- **Evaluation**:
  - Models are evaluated using metrics such as accuracy, precision, recall, F1-score, and confusion matrices.
- **Visualization**:
  - Plots of extracted features and confusion matrices are saved for analysis.

## Project Structure

README.txt
requirements.txt
run.py
data/
    augmented/
        augmented_features.csv
    processed/
        audio_features.csv
    raw/
        features_3_sec.csv
        features_30_sec.csv
        genres_original/
        images_original/
docs/
    report.md
    figures/
        blues/
        classical/
        country/
        ...
    test_results/
        knn_confusion_matrix.png
        svm_confusion_matrix.png
        random_forest_confusion_matrix.png
        cnn_confusion_matrix.png
logs/
    test_results.log
    training.log
models/
    checkpoint.pth
    cnn.h5
    knn_tuned.pkl
    random_forest_tuned.pkl
    saved_model.h5
    scaler.pkl
    svm_tuned.pkl
notebooks/
    MLproj (6).ipynb
src/
    data_processing/
        (Scripts for feature extraction, data augmentation, and preprocessing)
    evaluation/
        (Scripts for evaluating models and saving results)
    model/
        (Scripts forreation, training, and saving)
    utils/
        (Utility scripts such as configuration and logging)

## Key Files and Directories
- **`requirements.txt`**:
  - Contains the list of required Python libraries and their versions.
- **`run.py`**:
  - Main script to train and evaluate models.
- **`data/`**:
  - Contains raw, processed, and augmented datasets.
- **`docs/`**:
  - Includes project documentation, figures, and test results.
- **`logs/`**:
  - Stores logs for training, testing, and evaluation.
- **`models/`**:
  - Contains trained models and related artifacts (e.g., scaler, checkpoints).
- **`notebooks/`**:
  - Jupyter notebooks for exploratory data analysis (EDA) and model training.
- **`src/`**:
  - Contains the source code for data processing, evaluation, and utilities.

## How to Run the Project
1. **Install Dependencies**:
   - Install the required libraries using `requirements.txt`:
     ```sh
     pip install -r requirements.txt
     ```

2. **Prepare the Dataset**:
   - Place the raw audio files in the `data/raw/` directory.
   - Run the data augmentation script:
     ```sh
     python -m src.data_processing.augment
     ```

3. **Extract Features**:
   - Extract features from the audio files:
     ```sh
     python -m src.data_processing.extract_features
     ```

4. **Train Models**:
   - Train the machine learning and deep learning models:
     ```sh
     python run.py
     ```

5. **Evaluate Models**:
   - Evaluate the trained models and save results:
     ```sh
     python -m src.evaluation.test
     ```

6. **View Results**:
   - Check the `docs/test_results/` directory for evaluation metrics and confusion matrix plots.

## Dependencies
The project requires the following Python libraries:
- numpy==1.25.0
- pandas==2.1.0
- scikit-learn==1.3.0
- matplotlib==3.8.0
- seaborn==0.13.0
- tensorflow==2.13.0
- keras==2.13.0
- librosa==0.11.0
- audioread==3.0.1
- tqdm==4.68.0
- joblib==1.3.0

## Results
- **Best Model**: SVM achieved an accuracy of **99.60%**.
- **Confusion Matrices**:
  - Confusion matrices for each model are saved in the `docs/test_results/` directory.
- **Feature Visualizations**:
  - Plots of extracted features (e.g., Chroma STFT, RMS Energy, MFCCs) are saved in the `docs/figures/` directory.



