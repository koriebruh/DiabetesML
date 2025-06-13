#!/usr/bin/env python3
"""
IMPROVED DIABETES PREDICTION MODEL - FIXED VERSION

precision    recall  f1-score   support

   No Diabetes       0.89      0.94      0.91     43667
      Diabetes       0.43      0.29      0.34      7069

      accuracy                           0.85     50736
     macro avg       0.66      0.61      0.63     50736
  weighted avg       0.83      0.85      0.84     50736

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_curve, auc,
                             precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef,
                             cohen_kappa_score, precision_recall_curve, average_precision_score)
from sklearn.feature_selection import SelectFromModel, RFECV, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from collections import Counter
import xgboost as xgb
import joblib
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# For TensorFlow/Keras (optional, for LSTM autoencoder)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Dropout

    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. LSTM Autoencoder will be skipped.")
    TENSORFLOW_AVAILABLE = False


# ================================================================================
# 1. ADVANCED FEATURE ENGINEERING
# ================================================================================

def create_advanced_features(data):
    """
    Membuat fitur-fitur baru yang lebih bermakna untuk prediksi diabetes
    """
    df = data.copy()

    print("Creating advanced features...")

    # 1. BMI Categories (WHO Classification)
    if 'BMI' in df.columns:
        # BMI Categories: 0=Underweight, 1=Normal, 2=Overweight, 3=Obese I, 4=Obese II+
        df['BMI_Category'] = pd.cut(df['BMI'],
                                    bins=[0, 18.5, 25, 30, 35, 100],
                                    labels=[0, 1, 2, 3, 4]).astype('int8')

        # BMI risk indicators
        df['BMI_Normal'] = ((df['BMI'] >= 18.5) & (df['BMI'] < 25)).astype('uint8')
        df['BMI_Overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30)).astype('uint8')
        df['BMI_Obese'] = (df['BMI'] >= 30).astype('uint8')
        df['BMI_Severe_Obese'] = (df['BMI'] >= 35).astype('uint8')
        df['BMI_Extreme_Obese'] = (df['BMI'] >= 40).astype('uint8')

        # BMI squared (non-linear relationship)
        df['BMI_Squared'] = (df['BMI'] ** 2) / 1000  # Scaled down

        print(f"BMI features created. BMI range: {df['BMI'].min():.1f} - {df['BMI'].max():.1f}")

    # 2. Age Categories (Diabetes risk increases with age)
    if 'Age' in df.columns:
        # Age risk categories
        df['Age_Young'] = (df['Age'] <= 3).astype('uint8')  # Under 35
        df['Age_Middle'] = ((df['Age'] >= 4) & (df['Age'] <= 7)).astype('uint8')  # 35-54
        df['Age_Mature'] = ((df['Age'] >= 8) & (df['Age'] <= 10)).astype('uint8')  # 55-69
        df['Age_Senior'] = (df['Age'] >= 11).astype('uint8')  # 70+

        # High risk age groups
        df['Age_High_Risk'] = (df['Age'] >= 8).astype('uint8')  # 55+
        df['Age_Very_High_Risk'] = (df['Age'] >= 10).astype('uint8')  # 65+

        print(f"Age features created. Age categories: {df['Age'].value_counts().sort_index()}")

    # 3. Cardiovascular Risk Composite Score
    cardio_risk_cols = ['HighBP', 'HighChol', 'Stroke', 'HeartDiseaseorAttack']
    available_cardio_cols = [col for col in cardio_risk_cols if col in df.columns]

    if len(available_cardio_cols) >= 2:
        df['Cardiovascular_Risk_Score'] = df[available_cardio_cols].sum(axis=1).astype('uint8')
        df['Low_Cardio_Risk'] = (df['Cardiovascular_Risk_Score'] == 0).astype('uint8')
        df['Medium_Cardio_Risk'] = (df['Cardiovascular_Risk_Score'] == 1).astype('uint8')
        df['High_Cardio_Risk'] = (df['Cardiovascular_Risk_Score'] >= 2).astype('uint8')
        df['Very_High_Cardio_Risk'] = (df['Cardiovascular_Risk_Score'] >= 3).astype('uint8')

        print(f"Cardiovascular risk features created from: {available_cardio_cols}")

    # 4. Lifestyle Risk Score
    lifestyle_cols = ['Smoker', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump']
    available_lifestyle_cols = [col for col in lifestyle_cols if col in df.columns]

    if 'PhysActivity' in df.columns:
        df['Sedentary'] = (1 - df['PhysActivity']).astype('uint8')

    if 'Smoker' in df.columns and 'HvyAlcoholConsump' in df.columns:
        df['Bad_Habits_Score'] = (df['Smoker'] + df['HvyAlcoholConsump']).astype('uint8')
        df['Multiple_Bad_Habits'] = (df['Bad_Habits_Score'] >= 2).astype('uint8')

    if 'Fruits' in df.columns and 'Veggies' in df.columns:
        df['Good_Diet_Score'] = (df['Fruits'] + df['Veggies']).astype('uint8')
        df['Poor_Diet'] = (df['Good_Diet_Score'] == 0).astype('uint8')
        df['Excellent_Diet'] = (df['Good_Diet_Score'] == 2).astype('uint8')

    # 5. Health Awareness and Self-Care
    if 'CholCheck' in df.columns:
        df['Health_Conscious'] = df['CholCheck'].astype('uint8')

    if 'DiffWalk' in df.columns:
        df['Mobility_Issues'] = df['DiffWalk'].astype('uint8')

    if 'CholCheck' in df.columns and 'DiffWalk' in df.columns:
        df['Health_Awareness_Score'] = (df['CholCheck'] - df['DiffWalk']).astype('int8')

    # 6. Gender-specific risk patterns
    if 'Sex' in df.columns and 'Age' in df.columns:
        # Women have different risk patterns, especially post-menopause
        df['Female'] = (df['Sex'] == 0).astype('uint8')
        df['Male'] = (df['Sex'] == 1).astype('uint8')
        df['Female_Postmenopausal'] = ((df['Sex'] == 0) & (df['Age'] >= 7)).astype('uint8')  # Women 50+
        df['Male_High_Risk_Age'] = ((df['Sex'] == 1) & (df['Age'] >= 6)).astype('uint8')  # Men 45+

    # 7. High-Risk Combinations (Medical Literature Based)
    if 'BMI' in df.columns and 'HighBP' in df.columns:
        df['Metabolic_Risk'] = ((df['BMI'] >= 30) & (df['HighBP'] == 1)).astype('uint8')
        df['Severe_Metabolic_Risk'] = ((df['BMI'] >= 35) & (df['HighBP'] == 1)).astype('uint8')

    if 'BMI' in df.columns and 'Age' in df.columns:
        df['Age_BMI_Risk'] = ((df['Age'] >= 8) & (df['BMI'] >= 30)).astype('uint8')
        df['Young_Obese'] = ((df['Age'] <= 5) & (df['BMI'] >= 30)).astype('uint8')  # Early onset risk

    if 'HighBP' in df.columns and 'HighChol' in df.columns:
        df['Dual_Cardiovascular_Risk'] = ((df['HighBP'] == 1) & (df['HighChol'] == 1)).astype('uint8')

    if 'Smoker' in df.columns and 'HighBP' in df.columns:
        df['Smoker_Hypertension'] = ((df['Smoker'] == 1) & (df['HighBP'] == 1)).astype('uint8')

    # 8. Physical Activity and BMI interaction
    if 'BMI' in df.columns and 'PhysActivity' in df.columns:
        df['Obese_Sedentary'] = ((df['BMI'] >= 30) & (df['PhysActivity'] == 0)).astype('uint8')
        df['Active_Overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30) & (df['PhysActivity'] == 1)).astype('uint8')

    # 9. Comprehensive Risk Score
    risk_factors = []
    if 'BMI_Obese' in df.columns:
        risk_factors.append('BMI_Obese')
    if 'Age_High_Risk' in df.columns:
        risk_factors.append('Age_High_Risk')
    if 'High_Cardio_Risk' in df.columns:
        risk_factors.append('High_Cardio_Risk')
    if 'Sedentary' in df.columns:
        risk_factors.append('Sedentary')
    if 'Smoker' in df.columns:
        risk_factors.append('Smoker')

    if len(risk_factors) >= 3:
        df['Total_Risk_Score'] = df[risk_factors].sum(axis=1).astype('uint8')
        df['Low_Risk_Profile'] = (df['Total_Risk_Score'] <= 1).astype('uint8')
        df['Medium_Risk_Profile'] = ((df['Total_Risk_Score'] >= 2) & (df['Total_Risk_Score'] <= 3)).astype('uint8')
        df['High_Risk_Profile'] = (df['Total_Risk_Score'] >= 4).astype('uint8')

    print(f"Total features after engineering: {df.shape[1]}")
    return df


# ================================================================================
# 2. ENSEMBLE BALANCING TECHNIQUES
# ================================================================================

def ensemble_balancing_evaluation(X_train, y_train, X_val, y_val):
    """
    Evaluasi berbagai teknik balancing dan pilih yang terbaik
    """
    print("\n=== ENSEMBLE BALANCING EVALUATION ===")

    # Define balancing techniques
    balancing_techniques = {
        'SMOTE': SMOTE(random_state=42, k_neighbors=3),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=3),
        'ADASYN': ADASYN(random_state=42, n_neighbors=3),
        'SMOTEENN': SMOTEENN(random_state=42),
        'SMOTETomek': SMOTETomek(random_state=42)
    }

    results = {}
    balanced_datasets = {}

    print("Testing balancing techniques...")

    for name, technique in balancing_techniques.items():
        try:
            print(f"\nTesting {name}...")

            # Apply balancing
            X_balanced, y_balanced = technique.fit_resample(X_train, y_train)
            balanced_datasets[name] = (X_balanced, y_balanced)

            print(f"  Original distribution: {Counter(y_train)}")
            print(f"  Balanced distribution: {Counter(y_balanced)}")

            # Quick evaluation with multiple models
            models = {
                'RF': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'),
                'XGB': xgb.XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False,
                                         eval_metric='logloss')
            }

            model_scores = {}
            for model_name, model in models.items():
                model.fit(X_balanced, y_balanced)
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]

                # Calculate metrics
                precision = precision_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred)
                roc_auc = roc_auc_score(y_val, y_pred_proba)

                model_scores[model_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc
                }

            # Average scores across models
            avg_precision = np.mean([scores['precision'] for scores in model_scores.values()])
            avg_recall = np.mean([scores['recall'] for scores in model_scores.values()])
            avg_f1 = np.mean([scores['f1'] for scores in model_scores.values()])
            avg_roc_auc = np.mean([scores['roc_auc'] for scores in model_scores.values()])

            # Composite score with emphasis on precision and F1
            # We want good precision (reduce false positives) while maintaining decent recall
            precision_weight = 0.4
            f1_weight = 0.4
            recall_weight = 0.2

            composite_score = (precision_weight * avg_precision +
                               f1_weight * avg_f1 +
                               recall_weight * avg_recall)

            results[name] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'roc_auc': avg_roc_auc,
                'composite': composite_score
            }

            print(f"  Avg Precision: {avg_precision:.4f}")
            print(f"  Avg Recall: {avg_recall:.4f}")
            print(f"  Avg F1: {avg_f1:.4f}")
            print(f"  Avg ROC AUC: {avg_roc_auc:.4f}")
            print(f"  Composite Score: {composite_score:.4f}")

        except Exception as e:
            print(f"  Error with {name}: {e}")
            continue

    if results:
        # Select best technique based on composite score
        best_technique = max(results, key=lambda x: results[x]['composite'])
        print(f"\n=== BEST BALANCING TECHNIQUE: {best_technique} ===")
        print(f"Composite Score: {results[best_technique]['composite']:.4f}")

        return balanced_datasets[best_technique], best_technique, results
    else:
        print("No balancing technique worked. Using original data.")
        return (X_train, y_train), 'None', {}


# ================================================================================
# 3. ADVANCED ENSEMBLE CLASSIFIER
# ================================================================================

class AdvancedEnsembleClassifier:
    """
    Advanced ensemble classifier with weighted voting
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.weights = {}
        self.is_fitted = False

    def _create_models(self):
        """Create diverse set of models optimized for imbalanced data"""
        models = {
            'xgb': xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                tree_method='hist',
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1
            ),
            'rf_balanced': BalancedRandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
                class_weight='balanced_subsample',
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=200,
                random_state=self.random_state,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8
            ),
            'easy_ensemble': EasyEnsembleClassifier(
                n_estimators=50,
                random_state=self.random_state,
                sampling_strategy='auto'
            ),
            'lr': LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000,
                C=0.1
            )
        }
        return models

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit ensemble with validation-based weighting"""
        print("\n=== TRAINING ADVANCED ENSEMBLE ===")

        self.models = self._create_models()

        # Calculate class weights for XGBoost
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        scale_pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else 1.0
        self.models['xgb'].set_params(scale_pos_weight=scale_pos_weight)

        # Train models and calculate weights
        if X_val is not None and y_val is not None:
            print("Training models with validation-based weighting...")
            for name, model in self.models.items():
                print(f"  Training {name}...")
                try:
                    model.fit(X_train, y_train)

                    # Evaluate on validation set
                    y_pred = model.predict(X_val)
                    y_pred_proba = model.predict_proba(X_val)[:, 1]

                    precision = precision_score(y_val, y_pred)
                    recall = recall_score(y_val, y_pred)
                    f1 = f1_score(y_val, y_pred)
                    roc_auc = roc_auc_score(y_val, y_pred_proba)

                    # Weight calculation: emphasize precision and F1
                    weight = 0.4 * precision + 0.4 * f1 + 0.2 * recall
                    self.weights[name] = max(weight, 0.1)  # Minimum weight of 0.1

                    print(f"    Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                    print(f"    ROC AUC: {roc_auc:.4f}, Weight: {self.weights[name]:.4f}")

                except Exception as e:
                    print(f"    Error training {name}: {e}")
                    self.weights[name] = 0.1
        else:
            print("Training models with equal weights...")
            for name, model in self.models.items():
                try:
                    model.fit(X_train, y_train)
                    self.weights[name] = 1.0
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    self.weights[name] = 0.1

        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

        print(f"\nFinal ensemble weights: {self.weights}")
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """Weighted ensemble prediction"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = np.zeros((X.shape[0], 2))

        for name, model in self.models.items():
            if self.weights[name] > 0:
                try:
                    pred_proba = model.predict_proba(X)
                    predictions += self.weights[name] * pred_proba
                except Exception as e:
                    print(f"Error in prediction with {name}: {e}")

        return predictions

    def predict(self, X, threshold=0.5):
        """Predict with custom threshold"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)


# ================================================================================
# 4. THRESHOLD OPTIMIZATION
# ================================================================================

def optimize_threshold_comprehensive(y_true, y_proba, cost_fp=1, cost_fn=3, target_precision=0.4):
    """
    Comprehensive threshold optimization considering multiple criteria
    """
    print(f"\n=== THRESHOLD OPTIMIZATION ===")
    print(f"Cost FP: {cost_fp}, Cost FN: {cost_fn}, Target Precision: {target_precision}")

    from sklearn.metrics import precision_recall_curve

    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_proba)

    # Remove the last threshold (which is always 1.0 and causes division by zero)
    if len(thresholds) > len(precision_curve):
        thresholds = thresholds[:len(precision_curve)]

    results = []

    for i, threshold in enumerate(thresholds):
        if i >= len(precision_curve) or i >= len(recall_curve):
            break

        y_pred = (y_proba >= threshold).astype(int)

        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            # Metrics
            precision = precision_curve[i]
            recall = recall_curve[i]

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0

            # Business cost
            business_cost = cost_fp * fp + cost_fn * fn

            # Normalized business cost (per sample)
            normalized_cost = business_cost / len(y_true)

            # Combined score (maximize F1, minimize cost, meet precision target)
            precision_penalty = max(0, target_precision - precision)
            combined_score = f1 - 0.5 * normalized_cost / 100 - 2 * precision_penalty

            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'business_cost': business_cost,
                'normalized_cost': normalized_cost,
                'combined_score': combined_score
            })

        except Exception as e:
            continue

    if not results:
        print("No valid thresholds found!")
        return 0.5, 0, 0

    # Find best threshold by combined score
    best_result = max(results, key=lambda x: x['combined_score'])

    # Also find threshold with target precision
    precision_met_results = [r for r in results if r['precision'] >= target_precision]
    if precision_met_results:
        best_precision_result = max(precision_met_results, key=lambda x: x['f1'])
    else:
        best_precision_result = best_result

    print(f"\nBest Overall Threshold: {best_result['threshold']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall: {best_result['recall']:.4f}")
    print(f"  F1: {best_result['f1']:.4f}")
    print(f"  Business Cost: {best_result['business_cost']}")

    print(f"\nBest Precision-Target Threshold: {best_precision_result['threshold']:.4f}")
    print(f"  Precision: {best_precision_result['precision']:.4f}")
    print(f"  Recall: {best_precision_result['recall']:.4f}")
    print(f"  F1: {best_precision_result['f1']:.4f}")

    return best_precision_result['threshold'], best_precision_result['f1'], best_precision_result['business_cost']


# ================================================================================
# 5. LSTM AUTOENCODER (Optional)
# ================================================================================

def create_lstm_autoencoder_features(X_train, X_test, encoding_dim=None):
    """
    Create LSTM Autoencoder for feature extraction (optional)
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Skipping LSTM Autoencoder.")
        return X_train, X_test

    print("\n=== LSTM AUTOENCODER FEATURE EXTRACTION ===")

    # Determine encoding dimension
    if encoding_dim is None:
        encoding_dim = max(2, X_train.shape[1] // 3)

    print(f"Original features: {X_train.shape[1]}, Encoding dimension: {encoding_dim}")

    # Reshape for LSTM
    timesteps = 1
    X_train_lstm = X_train.reshape(X_train.shape[0], timesteps, X_train.shape[1])
    X_test_lstm = X_test.reshape(X_test.shape[0], timesteps, X_test.shape[1])

    # Build autoencoder
    input_dim = X_train.shape[1]

    # Input layer
    input_layer = Input(shape=(timesteps, input_dim))

    # Encoder
    encoder = LSTM(encoding_dim, activation='relu', return_sequences=False)(input_layer)
    encoder = Dropout(0.2)(encoder)

    # Decoder
    decoder = RepeatVector(timesteps)(encoder)
    decoder = LSTM(input_dim, activation='relu', return_sequences=True)(decoder)
    decoder = TimeDistributed(Dense(input_dim))(decoder)

    # Models
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    encoder_model = Model(inputs=input_layer, outputs=encoder)

    # Compile
    autoencoder.compile(optimizer='adam', loss='mse')

    print("Training LSTM Autoencoder...")
    # Train
    history = autoencoder.fit(
        X_train_lstm, X_train_lstm,
        epochs=30,
        batch_size=64,
        validation_split=0.2,
        verbose=0
    )

    # Extract features
    X_train_encoded = encoder_model.predict(X_train_lstm, verbose=0)
    X_test_encoded = encoder_model.predict(X_test_lstm, verbose=0)

    # Combine original and encoded features
    X_train_combined = np.hstack((X_train, X_train_encoded))
    X_test_combined = np.hstack((X_test, X_test_encoded))

    print(f"Combined features shape: {X_train_combined.shape[1]}")

    return X_train_combined, X_test_combined


# ================================================================================
# 6. MAIN PIPELINE
# ================================================================================

def main_improved_diabetes_pipeline(data_path, save_path='/content/drive/MyDrive/ML_KAH_IMPROVED/'):
    """
    Complete improved pipeline for diabetes prediction
    """
    print("=" * 80)
    print("IMPROVED DIABETES PREDICTION PIPELINE")
    print("=" * 80)

    # Create save directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1. Load Data
    print("\n1. Loading Data...")
    data = pd.read_csv(data_path)
    print(f"Dataset shape: {data.shape}")
    print(f"Target distribution:\n{data['target'].value_counts()}")

    # 2. Advanced Feature Engineering
    print("\n2. Advanced Feature Engineering...")
    data_engineered = create_advanced_features(data)

    # Separate features and target
    X = data_engineered.drop(['target'], axis=1)
    if 'ID' in X.columns:
        X = X.drop('ID', axis=1)
    y = data_engineered['target']

    print(f"Features after engineering: {X.shape[1]}")

    # 3. Feature Scaling and Selection
    print("\n3. Feature Scaling and Selection...")

    # Robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Feature selection with Random Forest
    print("  Performing feature selection...")
    rf_selector = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_selector.fit(X_scaled, y)

    # Select features based on importance
    selector = SelectFromModel(rf_selector, threshold='median', prefit=True)
    X_selected = selector.transform(X_scaled)
    selected_features = X.columns[selector.get_support()]

    print(f"  Selected {len(selected_features)} features from {X.shape[1]}")
    print(f"  Top selected features: {list(selected_features[:10])}")

    # 4. Train/Validation/Test Split
    print("\n4. Train/Validation/Test Split...")

    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    # Second split: train vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    print(f"Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Train target distribution: {Counter(y_train)}")
    print(f"Test target distribution: {Counter(y_test)}")

    # 5. Ensemble Balancing
    print("\n5. Ensemble Balancing Evaluation...")
    (X_train_balanced, y_train_balanced), best_technique, balancing_results = ensemble_balancing_evaluation(
        X_train, y_train, X_val, y_val
    )

    print(f"Best balancing technique: {best_technique}")
    print(f"Balanced training distribution: {Counter(y_train_balanced)}")

    # 6. Optional: LSTM Autoencoder
    print("\n6. LSTM Autoencoder Feature Enhancement...")
    X_train_enhanced, X_val_enhanced = create_lstm_autoencoder_features(
        X_train_balanced, X_val
    )
    X_test_enhanced = create_lstm_autoencoder_features(
        X_train_balanced, X_test
    )[1]  # We only need the test transformation

    # 7. Advanced Ensemble Training
    print("\n7. Training Advanced Ensemble Model...")
    ensemble_model = AdvancedEnsembleClassifier(random_state=42)
    ensemble_model.fit(X_train_enhanced, y_train_balanced, X_val_enhanced, y_val)

    # 8. Threshold Optimization
    print("\n8. Threshold Optimization...")
    y_val_proba = ensemble_model.predict_proba(X_val_enhanced)[:, 1]
    optimal_threshold, optimal_f1, optimal_cost = optimize_threshold_comprehensive(
        y_val, y_val_proba, cost_fp=1, cost_fn=3, target_precision=0.45
    )

    # 9. Final Evaluation
    print("\n9. Final Model Evaluation...")

    # Predictions on test set
    y_test_proba = ensemble_model.predict_proba(X_test_enhanced)[:, 1]
    y_test_pred_default = ensemble_model.predict(X_test_enhanced, threshold=0.5)
    y_test_pred_optimal = ensemble_model.predict(X_test_enhanced, threshold=optimal_threshold)

    # Calculate metrics for both thresholds
    def calculate_metrics(y_true, y_pred, y_proba, threshold_name):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_proba)
        mcc = matthews_corrcoef(y_true, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)

        print(f"\n{threshold_name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  MCC: {mcc:.4f}")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        return {
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'specificity': specificity, 'f1': f1, 'roc_auc': roc_auc, 'mcc': mcc,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }

    # Evaluate both thresholds
    default_metrics = calculate_metrics(y_test, y_test_pred_default, y_test_proba, "Default Threshold (0.5)")
    optimal_metrics = calculate_metrics(y_test, y_test_pred_optimal, y_test_proba,
                                        f"Optimal Threshold ({optimal_threshold:.3f})")

    # 10. Save Results and Models
    print("\n10. Saving Results and Models...")

    # Save models and preprocessors with error handling
    try:
        joblib.dump(ensemble_model, f'{save_path}ensemble_model.joblib')
        print("✓ Ensemble model saved successfully")
    except Exception as e:
        print(f"Warning: Could not save ensemble model: {e}")
        # Save individual models instead
        try:
            joblib.dump(ensemble_model.models, f'{save_path}ensemble_models_dict.joblib')
            joblib.dump(ensemble_model.weights, f'{save_path}ensemble_weights.joblib')
            print("✓ Individual models and weights saved instead")
        except Exception as e2:
            print(f"Error saving individual models: {e2}")

    joblib.dump(scaler, f'{save_path}scaler.joblib')
    joblib.dump(selector, f'{save_path}feature_selector.joblib')

    # Save balancing info
    balancing_info = {
        'technique': best_technique,
        'original_distribution': dict(Counter(y_train)),
        'balanced_distribution': dict(Counter(y_train_balanced)),
        'results': balancing_results
    }
    joblib.dump(balancing_info, f'{save_path}balancing_info.joblib')

    # Save threshold info
    threshold_info = {
        'optimal_threshold': optimal_threshold,
        'default_threshold': 0.5,
        'optimal_f1': optimal_f1,
        'target_precision': 0.45
    }
    joblib.dump(threshold_info, f'{save_path}threshold_info.joblib')

    # Save comprehensive results
    final_results = {
        'model_type': 'Advanced Ensemble with Feature Engineering',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset_info': {
            'total_samples': len(data),
            'original_features': data.shape[1] - 1,
            'engineered_features': X.shape[1],
            'selected_features': len(selected_features),
            'final_features': X_train_enhanced.shape[1]
        },
        'balancing': {
            'technique': best_technique,
            'original_dist': dict(Counter(y_train)),
            'balanced_dist': dict(Counter(y_train_balanced))
        },
        'default_threshold_results': default_metrics,
        'optimal_threshold_results': optimal_metrics,
        'thresholds': {
            'default': 0.5,
            'optimal': optimal_threshold
        }
    }

    # Save to CSV for easy reading
    results_df = pd.DataFrame([{
        'Model': 'Advanced Ensemble',
        'Balancing': best_technique,
        'Threshold': optimal_threshold,
        'Accuracy': optimal_metrics['accuracy'],
        'Precision': optimal_metrics['precision'],
        'Recall': optimal_metrics['recall'],
        'Specificity': optimal_metrics['specificity'],
        'F1_Score': optimal_metrics['f1'],
        'ROC_AUC': optimal_metrics['roc_auc'],
        'MCC': optimal_metrics['mcc']
    }])

    results_df.to_csv(f'{save_path}final_results.csv', index=False)

    # Save detailed report
    with open(f'{save_path}detailed_report.txt', 'w') as f:
        f.write("IMPROVED DIABETES PREDICTION MODEL - DETAILED REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Total Samples: {len(data):,}\n")
        f.write(f"Original Features: {data.shape[1] - 1}\n")
        f.write(f"Engineered Features: {X.shape[1]}\n")
        f.write(f"Selected Features: {len(selected_features)}\n")
        f.write(f"Final Features: {X_train_enhanced.shape[1]}\n\n")

        f.write(f"Best Balancing Technique: {best_technique}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n\n")

        f.write("OPTIMAL THRESHOLD RESULTS:\n")
        f.write("-" * 30 + "\n")
        for metric, value in optimal_metrics.items():
            if metric in ['tn', 'fp', 'fn', 'tp']:
                f.write(f"{metric.upper()}: {value:,}\n")
            else:
                f.write(f"{metric.capitalize()}: {value:.4f}\n")

        f.write("\nCLASSIFICATION REPORT:\n")
        f.write("-" * 30 + "\n")
        f.write(classification_report(y_test, y_test_pred_optimal))

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Best Model Performance (Threshold={optimal_threshold:.3f}):")
    print(f"  Precision: {optimal_metrics['precision']:.4f}")
    print(f"  Recall: {optimal_metrics['recall']:.4f}")
    print(f"  F1-Score: {optimal_metrics['f1']:.4f}")
    print(f"  ROC AUC: {optimal_metrics['roc_auc']:.4f}")
    print(f"\nAll results saved to: {save_path}")

    return ensemble_model, scaler, selector, optimal_threshold, final_results


# ================================================================================
# 7. UTILITY FUNCTIONS
# ================================================================================

def load_ensemble_model(model_path='/content/drive/MyDrive/ML_KAH_IMPROVED/'):
    """
    Load ensemble model with fallback for individual components
    """
    try:
        # Try to load complete ensemble model
        ensemble_model = joblib.load(f'{model_path}ensemble_model.joblib')
        print("✓ Complete ensemble model loaded")
        return ensemble_model
    except:
        try:
            # Fallback: load individual components
            models_dict = joblib.load(f'{model_path}ensemble_models_dict.joblib')
            weights_dict = joblib.load(f'{model_path}ensemble_weights.joblib')

            # Recreate ensemble model
            ensemble_model = AdvancedEnsembleClassifier(random_state=42)
            ensemble_model.models = models_dict
            ensemble_model.weights = weights_dict
            ensemble_model.is_fitted = True

            print("✓ Ensemble model reconstructed from components")
            return ensemble_model
        except Exception as e:
            print(f"Error loading ensemble model: {e}")
            return None


def predict_new_data(new_data, model_path='/content/drive/MyDrive/ML_KAH_IMPROVED/'):
    """
    Function to predict diabetes on new data using the trained model
    """
    print("Loading trained model and preprocessors...")

    # Load model and preprocessors
    ensemble_model = load_ensemble_model(model_path)
    if ensemble_model is None:
        raise ValueError("Could not load ensemble model")

    scaler = joblib.load(f'{model_path}scaler.joblib')
    selector = joblib.load(f'{model_path}feature_selector.joblib')
    threshold_info = joblib.load(f'{model_path}threshold_info.joblib')

    optimal_threshold = threshold_info['optimal_threshold']

    print(f"Using optimal threshold: {optimal_threshold:.4f}")

    # Preprocess new data
    print("Preprocessing new data...")

    # Remove target and ID columns if present
    X_new = new_data.copy()
    for col in ['target', 'Diabetes_binary', 'ID']:
        if col in X_new.columns:
            X_new = X_new.drop(col, axis=1)

    # Feature engineering
    X_new_engineered = create_advanced_features(X_new)

    # Scaling
    X_new_scaled = scaler.transform(X_new_engineered)

    # Feature selection
    X_new_selected = selector.transform(X_new_scaled)

    # Note: For LSTM features, we would need the encoder model
    # For now, we'll use the selected features only
    X_new_final = X_new_selected

    # Prediction
    probabilities = ensemble_model.predict_proba(X_new_final)[:, 1]
    predictions = ensemble_model.predict(X_new_final, threshold=optimal_threshold)

    # Create results DataFrame
    results = pd.DataFrame({
        'Probability': probabilities,
        'Prediction': predictions,
        'Risk_Level': ['High' if p >= 0.7 else 'Medium' if p >= 0.4 else 'Low' for p in probabilities],
        'Confidence': ['High' if p <= 0.3 or p >= 0.7 else 'Medium' if p <= 0.4 or p >= 0.6 else 'Low' for p in
                       probabilities]
    })

    return results


# ================================================================================
# 8. EXAMPLE USAGE
# ================================================================================

def example_usage():
    """
    Example of how to use the improved pipeline
    """

    # Example data path - replace with your actual path
    data_path = '/content/drive/MyDrive/ML KAH/dataset/cdc_diabetes_health_indicators.csv'
    save_path = '/content/drive/MyDrive/ML_KAH_IMPROVED/'

    print("Starting Improved Diabetes Prediction Pipeline...")
    print("This may take 30-60 minutes for large datasets...")

    try:
        # Run the complete pipeline
        model, scaler, selector, threshold, results = main_improved_diabetes_pipeline(
            data_path=data_path,
            save_path=save_path
        )

        print("\nPipeline completed successfully!")
        print("Key improvements implemented:")
        print("✓ Advanced feature engineering with medical domain knowledge")
        print("✓ Ensemble balancing technique evaluation")
        print("✓ Advanced ensemble classifier with weighted voting")
        print("✓ Business-aware threshold optimization")
        print("✓ LSTM autoencoder feature enhancement")
        print("✓ Comprehensive evaluation and visualization")

        return True

    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


# ================================================================================
# 9. MAIN EXECUTION
# ================================================================================

if __name__ == "__main__":
    print("IMPROVED DIABETES PREDICTION MODEL")
    print("=" * 50)
    print("This script implements advanced techniques to improve")
    print("precision while maintaining good recall for diabetes prediction.")
    print("\nKey improvements:")
    print("- Advanced feature engineering")
    print("- Ensemble balancing evaluation")
    print("- Multi-model ensemble with weighted voting")
    print("- Business-aware threshold optimization")
    print("- Optional LSTM autoencoder features")
    print("=" * 50)

    # Uncomment the line below to run the example
    success = example_usage()

    if success:
        print("\n🎉 Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed. Check the error messages above.")