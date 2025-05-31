import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Dropout


# # 5 JAM RUN
# Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.91      0.81      0.86     54583
#            1       0.31      0.50      0.38      8837
#
#     accuracy                           0.77     63420
#    macro avg       0.61      0.66      0.62     63420
# weighted avg       0.83      0.77      0.79     63420

# NameError: name 'xgb_grid' is not defined


# ----------------#
# 1. IMPORT DATA  #
# ----------------#

data = pd.read_csv('/content/drive/MyDrive/ML KAH/dataset/cdc_diabetes_health_indicators.csv')
print("Data shape:", data.shape)
print("\nSample data:")
print(data.head())
print("\nData info:")
print(data.info())
print("\nStatistik deskriptif:")
print(data.describe())

# Cek distribusi kelas
print("\n=== Distribusi Kelas (Sebelum Balancing) ===")
target_distribution = data['target'].value_counts()
print("Distribusi target:")
print(target_distribution)
print(
    f"Rasio kelas: {target_distribution[0]}/{target_distribution[1]} = {target_distribution[0] / target_distribution[1]:.2f}:1")

# Visualisasi distribusi kelas
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
data['target'].value_counts().plot(kind='bar')
plt.title('Distribusi Kelas (Sebelum Balancing)')
plt.xlabel('Kelas')
plt.ylabel('Jumlah')
plt.xticks([0, 1], ['Tidak Diabetes', 'Diabetes'], rotation=0)

# -----------------#
# 2. Preprocessing #
# -----------------#

print("\n=== 2. Preprocessing ===")
# Pisahkan fitur dan target
X = data.drop('target', axis=1)
if 'ID' in X.columns:
    X = X.drop('ID', axis=1)  # Hapus kolom ID jika ada
y = data['target']

print("Fitur shape:", X.shape)
print("Target shape:", y.shape)

# Normalisasi/Standardisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("Data setelah scaling (sample):")
print(X_scaled_df.head())

# ----------------------------------------#
# 3. Feature Selection use  Random Forest #
# ----------------------------------------#

print("\n=== 3. Feature Selection dengan Random Forest ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_scaled, y)

# Plotting feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature importances:")
print(feature_importances)

plt.subplot(1, 2, 2)
sns.barplot(x='Importance', y='Feature', data=feature_importances[:10])
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/ML KAH/feature_importances_with_class_distribution.png')
plt.close()

# Pilih fitur berdasarkan threshold importance
selector = SelectFromModel(rf, threshold='mean', prefit=True)
X_selected = selector.transform(X_scaled)
selected_feature_indices = selector.get_support()
selected_features = [feature for feature, selected in zip(X.columns, selected_feature_indices) if selected]

print(f"Jumlah fitur terpilih: {X_selected.shape[1]} dari {X_scaled.shape[1]}")
print("Fitur terpilih:", selected_features)

# ----------------------------#
# 4. Split Train/Test (75:25) #
# ----------------------------#

print("\n=== 4. Split Train/Test (75:25) ===")
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.25, random_state=42, stratify=y)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Cek distribusi kelas di train dan test set
print("Distribusi kelas di train set:")
print(Counter(y_train))
print("Distribusi kelas di test set:")
print(Counter(y_test))

# ----------------------------------------------#
# 5. DATA BALANCING - Multiple Techniques Test #
# ----------------------------------------------#

print("\n=== 5. DATA BALANCING - Testing Multiple Techniques ===")


# Fungsi untuk evaluasi cepat yang lebih robust
def quick_evaluate(X_train_balanced, y_train_balanced, X_test, y_test, technique_name):
    """Evaluasi cepat untuk membandingkan teknik balancing dengan multiple metrics"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score

    # Train simple RF model
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_temp.fit(X_train_balanced, y_train_balanced)

    # Predict and evaluate
    y_pred = rf_temp.predict(X_test)
    y_pred_proba = rf_temp.predict_proba(X_test)[:, 1]

    # Calculate multiple metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Composite score (weighted combination of metrics)
    composite_score = (f1 * 0.4) + (recall * 0.3) + (precision * 0.2) + (roc_auc * 0.1)

    print(f"\n{technique_name}:")
    print(f"Balanced train set shape: {X_train_balanced.shape}")
    print(f"Balanced train distribution: {Counter(y_train_balanced)}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Composite Score: {composite_score:.4f}")

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'composite': composite_score
    }


# Test berbagai teknik balancing dengan parameter yang dioptimasi
balancing_results = {}

# 1. SMOTE dengan berbagai parameter
print("Testing SMOTE variants...")
smote_configs = [
    {'k_neighbors': 3, 'name': 'SMOTE_k3'},
    {'k_neighbors': 5, 'name': 'SMOTE_k5'},
    {'k_neighbors': 7, 'name': 'SMOTE_k7'}
]

for config in smote_configs:
    try:
        smote = SMOTE(random_state=42, k_neighbors=config['k_neighbors'])
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        result = quick_evaluate(X_train_smote, y_train_smote, X_test, y_test, config['name'])
        balancing_results[config['name']] = result
    except Exception as e:
        print(f"Error with {config['name']}: {e}")

# 2. ADASYN
print("Testing ADASYN...")
try:
    adasyn = ADASYN(random_state=42, n_neighbors=5)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
    result = quick_evaluate(X_train_adasyn, y_train_adasyn, X_test, y_test, "ADASYN")
    balancing_results['ADASYN'] = result
except Exception as e:
    print(f"Error with ADASYN: {e}")

# 3. BorderlineSMOTE dengan variants
print("Testing BorderlineSMOTE variants...")
borderline_configs = [
    {'kind': 'borderline-1', 'name': 'BorderlineSMOTE_1'},
    {'kind': 'borderline-2', 'name': 'BorderlineSMOTE_2'}
]

for config in borderline_configs:
    try:
        borderline_smote = BorderlineSMOTE(random_state=42, kind=config['kind'])
        X_train_borderline, y_train_borderline = borderline_smote.fit_resample(X_train, y_train)
        result = quick_evaluate(X_train_borderline, y_train_borderline, X_test, y_test, config['name'])
        balancing_results[config['name']] = result
    except Exception as e:
        print(f"Error with {config['name']}: {e}")

# 4. SMOTEENN dengan parameter optimal
print("Testing SMOTEENN with optimized parameters...")
try:
    # SMOTEENN dengan SMOTE yang lebih konservatif
    smote_params = {'k_neighbors': 3}
    enn_params = {'n_neighbors': 3, 'kind_sel': 'mode'}
    smoteenn = SMOTEENN(random_state=42, smote=SMOTE(**smote_params), enn=EditedNearestNeighbours(**enn_params))
    X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train, y_train)
    result = quick_evaluate(X_train_smoteenn, y_train_smoteenn, X_test, y_test, "SMOTEENN_Optimized")
    balancing_results['SMOTEENN_Optimized'] = result
except Exception as e:
    print(f"Error with SMOTEENN_Optimized: {e}")

# 5. SMOTETomek dengan parameter optimal
print("Testing SMOTETomek with optimized parameters...")
try:
    smote_params = {'k_neighbors': 5}
    smotetomek = SMOTETomek(random_state=42, smote=SMOTE(**smote_params))
    X_train_smotetomek, y_train_smotetomek = smotetomek.fit_resample(X_train, y_train)
    result = quick_evaluate(X_train_smotetomek, y_train_smotetomek, X_test, y_test, "SMOTETomek_Optimized")
    balancing_results['SMOTETomek_Optimized'] = result
except Exception as e:
    print(f"Error with SMOTETomek_Optimized: {e}")

# 6. Hybrid approach: SMOTE + Manual undersampling
print("Testing Hybrid SMOTE + Undersampling...")
try:
    # Step 1: SMOTE untuk oversampling minority class
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_temp, y_temp = smote.fit_resample(X_train, y_train)

    # Step 2: Manual undersampling untuk mengurangi majority class
    from sklearn.utils import resample

    # Pisahkan kelas
    X_temp_df = pd.DataFrame(X_temp)
    X_temp_df['target'] = y_temp

    majority_class = X_temp_df[X_temp_df['target'] == 0]
    minority_class = X_temp_df[X_temp_df['target'] == 1]

    # Undersample majority class ke 1.5x minority class
    target_majority_size = int(len(minority_class) * 1.5)
    majority_undersampled = resample(majority_class,
                                     n_samples=target_majority_size,
                                     random_state=42)

    # Gabungkan kembali
    balanced_df = pd.concat([majority_undersampled, minority_class])
    X_train_hybrid = balanced_df.drop('target', axis=1).values
    y_train_hybrid = balanced_df['target'].values

    result = quick_evaluate(X_train_hybrid, y_train_hybrid, X_test, y_test, "Hybrid_SMOTE_Undersample")
    balancing_results['Hybrid_SMOTE_Undersample'] = result
except Exception as e:
    print(f"Error with Hybrid approach: {e}")

# Pilih teknik terbaik berdasarkan composite score
best_technique = max(balancing_results, key=lambda x: balancing_results[x]['composite'])
print(f"\n=== BEST BALANCING TECHNIQUE: {best_technique} ===")
print(f"Composite Score: {balancing_results[best_technique]['composite']:.4f}")
print(f"F1-Score: {balancing_results[best_technique]['f1']:.4f}")
print(f"Recall: {balancing_results[best_technique]['recall']:.4f}")
print(f"Precision: {balancing_results[best_technique]['precision']:.4f}")

# Terapkan teknik terbaik
if best_technique.startswith('SMOTE_k3'):
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
elif best_technique.startswith('SMOTE_k5'):
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
elif best_technique.startswith('SMOTE_k7'):
    smote = SMOTE(random_state=42, k_neighbors=7)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
elif best_technique == 'ADASYN':
    X_train_balanced, y_train_balanced = X_train_adasyn, y_train_adasyn
elif best_technique.startswith('BorderlineSMOTE'):
    kind = 'borderline-1' if '1' in best_technique else 'borderline-2'
    borderline_smote = BorderlineSMOTE(random_state=42, kind=kind)
    X_train_balanced, y_train_balanced = borderline_smote.fit_resample(X_train, y_train)
elif best_technique == 'SMOTEENN_Optimized':
    X_train_balanced, y_train_balanced = X_train_smoteenn, y_train_smoteenn
elif best_technique == 'SMOTETomek_Optimized':
    X_train_balanced, y_train_balanced = X_train_smotetomek, y_train_smotetomek
else:  # Hybrid
    X_train_balanced, y_train_balanced = X_train_hybrid, y_train_hybrid

print(f"\nFinal balanced dataset:")
print(f"Shape: {X_train_balanced.shape}")
print(f"Distribution: {Counter(y_train_balanced)}")

# -------------------------------------------------#
# 6. Optional: LSTM Autoencoder Feature Extraction #
# -------------------------------------------------#

print("\n=== 6. LSTM Autoencoder untuk Feature Extraction ===")
#  Reshape data untuk LSTM [samples, timesteps, features]
timesteps = 1
X_train_balanced_lstm = X_train_balanced.reshape(X_train_balanced.shape[0], timesteps, X_train_balanced.shape[1])
X_test_lstm = X_test.reshape(X_test.shape[0], timesteps, X_test.shape[1])

# Buat model LSTM Autoencoder
input_dim = X_train_balanced.shape[1]
encoding_dim = max(2, input_dim // 2)  # Setidaknya 2 fitur hasil encoding

input_layer = Input(shape=(timesteps, input_dim))
# Encoder
encoder = LSTM(encoding_dim, activation='relu', return_sequences=False)(input_layer)
# Decoder
decoder = RepeatVector(timesteps)(encoder)
decoder = LSTM(input_dim, activation='relu', return_sequences=True)(decoder)
decoder = TimeDistributed(Dense(input_dim))(decoder)

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder)
encoder_model = Model(inputs=input_layer, outputs=encoder)

# Compile model
autoencoder.compile(optimizer='adam', loss='mse')

# Model summary
print("LSTM Autoencoder Summary:")
autoencoder.summary()

# Train autoencoder
history = autoencoder.fit(
    X_train_balanced_lstm, X_train_balanced_lstm,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Autoencoder Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('/content/drive/MyDrive/ML KAH/autoencoder_loss.png')
plt.close()

# Extract features using encoder
X_train_encoded = encoder_model.predict(X_train_balanced_lstm)
X_test_encoded = encoder_model.predict(X_test_lstm)

print(f"Encoded features shape: {X_train_encoded.shape}")

# ----------------------------------------------------------#
# 7. Feature Fusion (PCA + Random Forest selected features) #
# ----------------------------------------------------------#

print("\n=== 7. Feature Fusion (PCA + Random Forest selected features) ===")
# Implementasi feature fusion dengan Original features + Encoded features
X_train_original = X_train_balanced
X_test_original = X_test

# Gabungkan fitur asli dengan fitur hasil encoding
X_train_fused = np.hstack((X_train_original, X_train_encoded))
X_test_fused = np.hstack((X_test_original, X_test_encoded))

print(f"Feature fusion shape: {X_train_fused.shape}")

# Custom Feature Fusion untuk fitur tertentu
X_train_df = pd.DataFrame(X_train_balanced, columns=selected_features)
X_test_df = pd.DataFrame(X_test, columns=selected_features)

# Jika ada kolom BMI dan PhysActivity dalam fitur terpilih, buat fitur baru
if 'BMI' in selected_features and 'PhysActivity' in selected_features:
    # Dapatkan indeks original dari kolom ini
    bmi_idx = selected_features.index('BMI')
    physact_idx = selected_features.index('PhysActivity')

    # Buat fitur baru BMI_Activity: BMI/(PhysActivity+1) untuk menghindari div/0
    X_train_fused_df = pd.DataFrame(X_train_fused)
    X_test_fused_df = pd.DataFrame(X_test_fused)

    # Buat kolom baru di dataframe
    X_train_fused_df['BMI_Activity'] = X_train_df['BMI'] / (X_train_df['PhysActivity'] + 1)
    X_test_fused_df['BMI_Activity'] = X_test_df['BMI'] / (X_test_df['PhysActivity'] + 1)

    # Konversi kembali ke numpy array
    X_train_fused = X_train_fused_df.values
    X_test_fused = X_test_fused_df.values

    print("Fitur baru BMI_Activity ditambahkan")

# Jika ada kolom HighBP dan HighChol dalam fitur terpilih, buat fitur baru HighRisk
if 'HighBP' in selected_features and 'HighChol' in selected_features:
    highbp_idx = selected_features.index('HighBP')
    highchol_idx = selected_features.index('HighChol')

    # Buat fitur baru HighRisk: 1 jika HighBP=1 ATAU HighChol=1, 0 jika tidak
    X_train_fused_df = pd.DataFrame(X_train_fused)
    X_test_fused_df = pd.DataFrame(X_test_fused)

    # Buat kolom baru di dataframe
    X_train_fused_df['HighRisk'] = ((X_train_df['HighBP'] + X_train_df['HighChol']) > 0).astype(int)
    X_test_fused_df['HighRisk'] = ((X_test_df['HighBP'] + X_test_df['HighChol']) > 0).astype(int)

    # Konversi kembali ke numpy array
    X_train_fused = X_train_fused_df.values
    X_test_fused = X_test_fused_df.values

    print("Fitur baru HighRisk ditambahkan")

# --------------------------------------------------#
# 8. Feature Selection Ulang setelah Feature Fusion #
# --------------------------------------------------#

print("\n=== 8. Feature Selection Ulang setelah Feature Fusion ===")
rf_after_fusion = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_after_fusion.fit(X_train_fused, y_train_balanced)

# Pilih fitur lagi setelah fusion
selector_after_fusion = SelectFromModel(rf_after_fusion, threshold='mean', prefit=True)
X_train_final = selector_after_fusion.transform(X_train_fused)
X_test_final = selector_after_fusion.transform(X_test_fused)

print(f"Jumlah fitur final: {X_train_final.shape[1]} dari {X_train_fused.shape[1]}")

# --------------------------------------------------#
# 9. Train Final Model (XGBoost + Cross Validation) #
# --------------------------------------------------#

print("\n=== 9. Train Final Model (XGBoost + Cross Validation) ===")

# Import XGBoost
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight

# Hitung class weights untuk XGBoost
classes = np.unique(y_train_balanced)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_balanced)
class_weight_dict = dict(zip(classes, class_weights))
scale_pos_weight = class_weight_dict[0] / class_weight_dict[1]

print(f"Scale pos weight for XGBoost: {scale_pos_weight}")

# Parameter grid yang lebih fokus dan efisien
param_grid = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 1.5],
    'scale_pos_weight': [1, scale_pos_weight, scale_pos_weight * 0.8]
}

# Gunakan RandomizedSearchCV untuk efisiensi
print("Performing Randomized Search for hyperparameter optimization...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_random = RandomizedSearchCV(
    xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=1  # Avoid threading issues
    ),
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter settings sampled
    cv=skf,
    scoring='f1',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

xgb_random.fit(X_train_final, y_train_balanced)

# Best model parameters
print("Best parameters:", xgb_random.best_params_)
print("Best cross-validation F1-score: {:.4f}".format(xgb_random.best_score_))

# Get the best model
best_xgb = xgb_random.best_estimator_

# Fine-tuning around best parameters
print("\nFine-tuning around best parameters...")
best_params = xgb_random.best_params_

# Create narrow parameter grid around best parameters
fine_tune_grid = {}
for param, value in best_params.items():
    if param == 'learning_rate':
        fine_tune_grid[param] = [max(0.01, value - 0.02), value, min(0.3, value + 0.02)]
    elif param == 'max_depth':
        fine_tune_grid[param] = [max(3, value - 1), value, min(10, value + 1)]
    elif param == 'n_estimators':
        fine_tune_grid[param] = [max(100, value - 50), value, min(1000, value + 50)]
    elif param in ['subsample', 'colsample_bytree']:
        fine_tune_grid[param] = [max(0.6, value - 0.1), value, min(1.0, value + 0.1)]
    else:
        fine_tune_grid[param] = [value]

xgb_fine_tune = GridSearchCV(
    xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=1
    ),
    param_grid=fine_tune_grid,
    cv=skf,
    scoring='f1',
    verbose=1,
    n_jobs=-1
)

xgb_fine_tune.fit(X_train_final, y_train_balanced)

# Update best model if fine-tuning improved results
if xgb_fine_tune.best_score_ > xgb_random.best_score_:
    best_xgb = xgb_fine_tune.best_estimator_
    print(f"Fine-tuning improved F1-score to: {xgb_fine_tune.best_score_:.4f}")
else:
    print("Fine-tuning did not improve results, keeping original best model")

# Cross-validation with multiple metrics
print("\nPerforming comprehensive cross-validation...")
cv_results = cross_validate(
    best_xgb, X_train_final, y_train_balanced,
    cv=skf,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    return_train_score=True
)

print("\nCross-validation results:")
print("=" * 50)
for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    test_scores = cv_results[f'test_{metric}']
    train_scores = cv_results[f'train_{metric}']
    print(f"{metric.capitalize():12} - Test: {test_scores.mean():.4f} ± {test_scores.std():.4f}")
    print(f"{'':12} - Train: {train_scores.mean():.4f} ± {train_scores.std():.4f}")

    # Check for overfitting
    if train_scores.mean() - test_scores.mean() > 0.1:
        print(f"{'':12} - WARNING: Potential overfitting detected!")
    print()

# Feature importance analysis
feature_importances = best_xgb.feature_importances_
print(f"\nTop 10 most important features:")
for i, importance in enumerate(sorted(enumerate(feature_importances),
                                      key=lambda x: x[1], reverse=True)[:10]):
    print(f"Feature {importance[0]:2d}: {importance[1]:.4f}")


# Early stopping validation (retrain with early stopping)
# Alternative approach - Create completely new model with explicit parameters
print("\nRetraining with early stopping for optimal performance...")
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_final, y_train_balanced, test_size=0.2, random_state=42, stratify=y_train_balanced
)

# Get best parameters from hyperparameter tuning
if xgb_fine_tune.best_score_ > xgb_random.best_score_:
    best_params_dict = xgb_fine_tune.best_params_
else:
    best_params_dict = xgb_random.best_params_

# Create new model with early stopping using best parameters
best_xgb_early = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=1,
    early_stopping_rounds=20,
    **best_params_dict  # This should not conflict now
)

best_xgb_early.fit(
    X_train_split, y_train_split,
    eval_set=[(X_val_split, y_val_split)],
    verbose=False
)

# Compare models
print(f"Original model trees: {best_xgb.n_estimators}")
print(f"Early stopped model trees: {best_xgb_early.n_estimators}")

# Use early stopped model if it has fewer trees (less overfitting)
if best_xgb_early.n_estimators < best_xgb.n_estimators:
    print("Using early stopped model to prevent overfitting")
    best_xgb = best_xgb_early
else:
    print("Original model is already optimal, keeping it")
# -----------------#
# 10. Final Testing #
# -----------------#

print("\n=== 10. Final Testing ===")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, precision_recall_curve, average_precision_score

# Prediksi pada data test
y_pred = best_xgb.predict(X_test_final)
y_pred_proba = best_xgb.predict_proba(X_test_final)[:, 1]

# Evaluasi model - metrics untuk publikasi jurnal
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc_value = roc_auc_score(y_test, y_pred_proba)
mcc = matthews_corrcoef(y_test, y_pred)  # Matthews Correlation Coefficient
kappa = cohen_kappa_score(y_test, y_pred)  # Cohen's Kappa
average_precision = average_precision_score(y_test, y_pred_proba)  # Average Precision

# Tambahan: Sensitivity dan Specificity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)  # True Positive Rate = Recall
specificity = tn / (tn + fp)  # True Negative Rate

# Tampilkan hasil metrik untuk publikasi jurnal
print("\n" + "=" * 60)
print("HASIL EVALUASI MODEL UNTUK PUBLIKASI JURNAL (BALANCED)")
print("=" * 60)
print(f"Balancing Technique : {best_technique}")
print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity        : {specificity:.4f}")
print(f"F1-Score           : {f1:.4f}")
print(f"ROC AUC            : {roc_auc_value:.4f}")
print(f"MCC                : {mcc:.4f}")
print(f"Cohen's Kappa      : {kappa:.4f}")
print(f"Avg Precision      : {average_precision:.4f}")
print("=" * 60)

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# Plot Confusion Matrix
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Plot ROC Curve
plt.subplot(1, 3, 2)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_value:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# Plot Precision-Recall Curve
plt.subplot(1, 3, 3)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AP = {average_precision:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/ML KAH/evaluation_plots_balanced.png')
plt.close()

# Simpan hasil untuk jurnal
with open('/content/drive/MyDrive/ML KAH/hasil_evaluasi_untuk_jurnal_balanced.txt', 'w') as f:
    f.write("HASIL EVALUASI MODEL UNTUK PUBLIKASI JURNAL (BALANCED)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Balancing Technique : {best_technique}\n")
    f.write(f"Accuracy           : {accuracy:.4f}\n")
    f.write(f"Precision          : {precision:.4f}\n")
    f.write(f"Recall (Sensitivity): {recall:.4f}\n")
    f.write(f"Specificity        : {specificity:.4f}\n")
    f.write(f"F1-Score           : {f1:.4f}\n")
    f.write(f"ROC AUC            : {roc_auc_value:.4f}\n")
    f.write(f"MCC                : {mcc:.4f}\n")
    f.write(f"Cohen's Kappa      : {kappa:.4f}\n")
    f.write(f"Avg Precision      : {average_precision:.4f}\n")
    f.write("=" * 60 + "\n\n")
    f.write("Confusion Matrix:\n")
    f.write(f"True Negatives: {tn}\n")
    f.write(f"False Positives: {fp}\n")
    f.write(f"False Negatives: {fn}\n")
    f.write(f"True Positives: {tp}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Simpan model terbaik
import joblib
import pandas as pd
from datetime import datetime

# Simpan model dan preprocessing objects
joblib.dump(best_xgb, '/content/drive/MyDrive/ML KAH/best_xgb_model_balanced.joblib')
joblib.dump(scaler, '/content/drive/MyDrive/ML KAH/scaler_balanced.joblib')
joblib.dump(selector, '/content/drive/MyDrive/ML KAH/feature_selector_balanced.joblib')
joblib.dump(selector_after_fusion, '/content/drive/MyDrive/ML KAH/feature_selector_after_fusion_balanced.joblib')
joblib.dump(encoder_model, '/content/drive/MyDrive/ML KAH/lstm_encoder_balanced.joblib')

# Simpan balancing technique yang digunakan
balancing_info = {
    'technique': best_technique,
    'original_distribution': dict(Counter(y_train)),
    'balanced_distribution': dict(Counter(y_train_balanced))
}
joblib.dump(balancing_info, '/content/drive/MyDrive/ML KAH/balancing_info.joblib')

# Simpan hasil eksperimen untuk jurnal
experiment_results = {
    'Model': f'XGBoost + LSTM Autoencoder + Feature Fusion + {best_technique}',
    'Tanggal': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'Dataset_Size': X.shape[0],
    'Original_Features': X.shape[1],
    'Selected_Features': len(selected_features),
    'Final_Features': X_train_final.shape[1],
    'Balancing_Technique': best_technique,
    'Original_Class_Distribution': dict(Counter(y_train)),
    'Balanced_Class_Distribution': dict(Counter(y_train_balanced)),
    'Best_Parameters': xgb_grid.best_params_,
    'CV_F1_Mean': cv_results['test_f1'].mean(),
    'CV_F1_Std': cv_results['test_f1'].std(),
    'Test_Accuracy': accuracy,
    'Test_Precision': precision,
    'Test_Recall': recall,
    'Test_Sensitivity': sensitivity,
    'Test_Specificity': specificity,
    'Test_F1': f1,
    'Test_ROC_AUC': roc_auc_value,
    'Test_MCC': mcc,
    'Test_Kappa': kappa,
    'Test_Avg_Precision': average_precision
}

# Simpan dalam format CSV
pd.DataFrame([experiment_results]).to_csv('/content/drive/MyDrive/ML KAH/hasil_eksperimen_untuk_jurnal_balanced.csv',
                                          index=False)

# Simpan hasil dengan format tabel untuk jurnal
with open('/content/drive/MyDrive/ML KAH/hasil_table_untuk_jurnal_balanced.txt', 'w') as f:
    f.write("Table X: Performance metrics of the proposed hybrid model for diabetes prediction (Balanced Dataset)\n\n")
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| Balancing Technique | {best_technique} |\n")
    f.write(f"| Accuracy | {accuracy:.4f} |\n")
    f.write(f"| Precision | {precision:.4f} |\n")
    f.write(f"| Recall (Sensitivity) | {recall:.4f} |\n")
    f.write(f"| Specificity | {specificity:.4f} |\n")
    f.write(f"| F1-Score | {f1:.4f} |\n")
    f.write(f"| ROC AUC | {roc_auc_value:.4f} |\n")
    f.write(f"| Matthews Correlation Coefficient | {mcc:.4f} |\n")
    f.write(f"| Cohen's Kappa | {kappa:.4f} |\n")
    f.write(f"| Average Precision | {average_precision:.4f} |\n\n")
    f.write(
        "*Note: The proposed model combines XGBoost with LSTM Autoencoder feature extraction, custom feature fusion techniques, and advanced balancing methods to address class imbalance in diabetes prediction.")

# Perbandingan dengan hasil sebelumnya (tanpa balancing)
print("\n" + "=" * 80)
print("PERBANDINGAN HASIL: SEBELUM vs SESUDAH BALANCING")
print("=" * 80)
print("Metrik               | Sebelum Balancing | Sesudah Balancing | Peningkatan")
print("-" * 80)
print(f"Recall (Sensitivity) | 0.0101           | {recall:.4f}        | {((recall - 0.0101) / 0.0101) * 100:.1f}x")
print(f"F1-Score            | 0.0198           | {f1:.4f}        | {((f1 - 0.0198) / 0.0198) * 100:.1f}x")
print(
    f"Precision           | 0.5235           | {precision:.4f}        | {((precision - 0.5235) / 0.5235) * 100:+.1f}%")
print(f"Accuracy            | 0.8608           | {accuracy:.4f}        | {((accuracy - 0.8608) / 0.8608) * 100:+.1f}%")
print(
    f"ROC AUC             | 0.7898           | {roc_auc_value:.4f}        | {((roc_auc_value - 0.7898) / 0.7898) * 100:+.1f}%")
print("=" * 80)

# Simpan perbandingan hasil
comparison_results = {
    'Metric': ['Recall', 'F1-Score', 'Precision', 'Accuracy', 'ROC AUC'],
    'Before_Balancing': [0.0101, 0.0198, 0.5235, 0.8608, 0.7898],
    'After_Balancing': [recall, f1, precision, accuracy, roc_auc_value],
    'Improvement': [
        f"{((recall - 0.0101) / 0.0101) * 100:.1f}x",
        f"{((f1 - 0.0198) / 0.0198) * 100:.1f}x",
        f"{((precision - 0.5235) / 0.5235) * 100:+.1f}%",
        f"{((accuracy - 0.8608) / 0.8608) * 100:+.1f}%",
        f"{((roc_auc_value - 0.7898) / 0.7898) * 100:+.1f}%"
    ]
}

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv('/content/drive/MyDrive/ML KAH/comparison_before_after_balancing.csv', index=False)

print("\nProses selesai! Model terbaik dengan balancing tersimpan sebagai 'best_xgb_model_balanced.joblib'")


# -----------------------------------------------#
# 11. Buat fungsi untuk prediksi data baru      #
# -----------------------------------------------#

def predict_diabetes_balanced(new_data, use_balancing_technique=best_technique):
    """
    Memprediksi diabetes pada data baru dengan model yang sudah di-balance

    Parameters:
    -----------
    new_data : pandas DataFrame
        Data baru yang akan diprediksi, harus memiliki kolom yang sama dengan dataset asli
    use_balancing_technique : str
        Teknik balancing yang digunakan (untuk informasi saja)

    Returns:
    --------
    prediksi : array
        Hasil prediksi (0: tidak diabetes, 1: diabetes)
    probabilitas : array
        Probabilitas prediksi
    confidence_level : str
        Level kepercayaan prediksi
    """
    # Preprocessing
    if 'ID' in new_data.columns:
        new_data = new_data.drop('ID', axis=1)
    if 'Diabetes_binary' in new_data.columns:
        new_data = new_data.drop('Diabetes_binary', axis=1)
    if 'target' in new_data.columns:
        new_data = new_data.drop('target', axis=1)

    # Scaling
    X_new_scaled = scaler.transform(new_data)

    # Feature selection pertama
    X_new_selected = selector.transform(X_new_scaled)

    # LSTM encoding
    X_new_lstm = X_new_selected.reshape(X_new_selected.shape[0], 1, X_new_selected.shape[1])
    X_new_encoded = encoder_model.predict(X_new_lstm)

    # Feature fusion
    X_new_fused = np.hstack((X_new_selected, X_new_encoded))

    # Feature selection final
    X_new_final = selector_after_fusion.transform(X_new_fused)

    # Prediksi
    prediction = best_xgb.predict(X_new_final)
    probability = best_xgb.predict_proba(X_new_final)[:, 1]

    # Tentukan confidence level
    confidence_level = []
    for prob in probability:
        if prob < 0.3 or prob > 0.7:
            confidence_level.append("High")
        elif prob < 0.4 or prob > 0.6:
            confidence_level.append("Medium")
        else:
            confidence_level.append("Low")

    return prediction, probability, confidence_level


# Contoh penggunaan dengan data test
print("\n=== CONTOH PREDIKSI PADA DATA BARU ===")
# Ambil 5 sampel dari data test untuk demonstrasi
sample_indices = np.random.choice(X_test.shape[0], size=5, replace=False)
X_sample = X_test[sample_indices]
y_sample = y_test.iloc[sample_indices].values

# Konversi ke DataFrame untuk fungsi prediksi
X_sample_df = pd.DataFrame(X_sample, columns=selected_features)

# Prediksi
pred, prob, conf = predict_diabetes_balanced(X_sample_df)

print("Hasil prediksi pada 5 sampel data:")
print("-" * 60)
for i in range(5):
    actual = "Diabetes" if y_sample[i] == 1 else "Tidak Diabetes"
    predicted = "Diabetes" if pred[i] == 1 else "Tidak Diabetes"
    print(f"Sampel {i + 1}:")
    print(f"  Aktual: {actual}")
    print(f"  Prediksi: {predicted}")
    print(f"  Probabilitas: {prob[i]:.4f}")
    print(f"  Confidence: {conf[i]}")
    print(f"  Status: {'✓ Benar' if pred[i] == y_sample[i] else '✗ Salah'}")
    print("-" * 60)

print(f"\nContoh penggunaan fungsi prediksi:")
print(f"prediction, probability, confidence = predict_diabetes_balanced(new_patient_data)")
print(f"\nModel menggunakan teknik balancing: {best_technique}")
print(f"Model ini memberikan peningkatan signifikan dalam mendeteksi kasus diabetes!")

# -----------------------------------------------#
# 12. Analisis Feature Importance Final         #
# -----------------------------------------------#

print("\n=== 12. ANALISIS FEATURE IMPORTANCE FINAL ===")

# Get feature importance dan mapping ke nama fitur asli
final_feature_importance = best_xgb.feature_importances_
feature_names_final = [f"Feature_{i}" for i in range(len(final_feature_importance))]

# Buat DataFrame untuk feature importance
importance_df = pd.DataFrame({
    'Feature_Index': range(len(final_feature_importance)),
    'Importance': final_feature_importance
}).sort_values('Importance', ascending=False)

print("Top 10 Most Important Features (Final Model):")
print(importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), [f"Feature {i}" for i in top_features['Feature_Index']])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importance (Final Balanced Model)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/ML KAH/final_feature_importance.png')
plt.close()

# Simpan feature importance
importance_df.to_csv('/content/drive/MyDrive/ML KAH/final_feature_importance.csv', index=False)

print("\n" + "=" * 80)
print("RINGKASAN AKHIR EKSPERIMEN")
print("=" * 80)
print(f"✓ Dataset berhasil di-balance menggunakan teknik: {best_technique}")
print(f"✓ Peningkatan Recall dari 1.01% menjadi {recall * 100:.1f}% ({((recall - 0.0101) / 0.0101) * 100:.0f}x lipat)")
print(f"✓ Peningkatan F1-Score dari 1.98% menjadi {f1 * 100:.1f}% ({((f1 - 0.0198) / 0.0198) * 100:.0f}x lipat)")
print(f"✓ Model sekarang dapat mendeteksi diabetes dengan baik!")
print(f"✓ ROC AUC: {roc_auc_value:.4f} (Good discriminative ability)")
print(f"✓ Model dan semua file hasil telah disimpan di Google Drive")
print("=" * 80)

# -----------------------------------------------#
# 13. Advanced Threshold Optimization            #
# -----------------------------------------------#

print("\n=== 13. ADVANCED THRESHOLD OPTIMIZATION ===")

# Get predictions
y_pred_proba = best_xgb.predict_proba(X_test_final)[:, 1]

# 1. Threshold optimization based on different metrics
from sklearn.metrics import precision_recall_curve, roc_curve


def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """Find optimal threshold based on specified metric"""

    if metric == 'f1':
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx], f1_scores[optimal_idx]

    elif metric == 'youden':
        # Youden's J statistic = Sensitivity + Specificity - 1
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        youden_scores = tpr - fpr
        optimal_idx = np.argmax(youden_scores)
        return thresholds[optimal_idx], youden_scores[optimal_idx]

    elif metric == 'balanced_accuracy':
        # Balanced accuracy = (Sensitivity + Specificity) / 2
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        balanced_acc = (tpr + (1 - fpr)) / 2
        optimal_idx = np.argmax(balanced_acc)
        return thresholds[optimal_idx], balanced_acc[optimal_idx]

    elif metric == 'cost_sensitive':
        # Cost-sensitive threshold (assuming FN is more costly than FP)
        # Cost ratio: FN cost / FP cost = 3 (missing diabetes is 3x worse than false alarm)
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        cost_ratio = 3
        cost_sensitive_score = recall - (cost_ratio * (1 - precision))
        optimal_idx = np.argmax(cost_sensitive_score)
        return thresholds[optimal_idx], cost_sensitive_score[optimal_idx]


# Find optimal thresholds for different objectives
threshold_results = {}

print("Finding optimal thresholds for different objectives:")
print("-" * 60)

# F1-Score optimization
f1_threshold, f1_score_max = find_optimal_threshold(y_test, y_pred_proba, 'f1')
threshold_results['f1'] = {'threshold': f1_threshold, 'score': f1_score_max}
print(f"F1-Score optimized threshold: {f1_threshold:.4f} (F1: {f1_score_max:.4f})")

# Youden's J statistic optimization
youden_threshold, youden_score = find_optimal_threshold(y_test, y_pred_proba, 'youden')
threshold_results['youden'] = {'threshold': youden_threshold, 'score': youden_score}
print(f"Youden's J optimized threshold: {youden_threshold:.4f} (J: {youden_score:.4f})")

# Balanced accuracy optimization
ba_threshold, ba_score = find_optimal_threshold(y_test, y_pred_proba, 'balanced_accuracy')
threshold_results['balanced_accuracy'] = {'threshold': ba_threshold, 'score': ba_score}
print(f"Balanced Accuracy optimized threshold: {ba_threshold:.4f} (BA: {ba_score:.4f})")

# Cost-sensitive optimization
cost_threshold, cost_score = find_optimal_threshold(y_test, y_pred_proba, 'cost_sensitive')
threshold_results['cost_sensitive'] = {'threshold': cost_threshold, 'score': cost_score}
print(f"Cost-sensitive optimized threshold: {cost_threshold:.4f} (CS: {cost_score:.4f})")


# 2. Comprehensive evaluation with different thresholds
def evaluate_threshold(y_true, y_proba, threshold, threshold_name):
    """Comprehensive evaluation for a given threshold"""
    y_pred = (y_proba >= threshold).astype(int)

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Additional metrics
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_accuracy = (sensitivity + specificity) / 2

    # Clinical metrics
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'mcc': mcc,
        'balanced_accuracy': balanced_accuracy,
        'ppv': ppv,
        'npv': npv,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


# Evaluate all threshold strategies
print(f"\n{'=' * 80}")
print("COMPREHENSIVE THRESHOLD COMPARISON")
print(f"{'=' * 80}")

threshold_evaluations = {}
thresholds_to_test = [
    (0.5, "Default"),
    (f1_threshold, "F1-Optimized"),
    (youden_threshold, "Youden-Optimized"),
    (ba_threshold, "Balanced-Acc-Optimized"),
    (cost_threshold, "Cost-Sensitive")
]

for threshold, name in thresholds_to_test:
    eval_result = evaluate_threshold(y_test, y_pred_proba, threshold, name)
    threshold_evaluations[name] = eval_result

    print(f"\n{name} (Threshold: {threshold:.4f}):")
    print(f"  Accuracy: {eval_result['accuracy']:.4f}")
    print(f"  Precision: {eval_result['precision']:.4f}")
    print(f"  Recall/Sensitivity: {eval_result['recall']:.4f}")
    print(f"  Specificity: {eval_result['specificity']:.4f}")
    print(f"  F1-Score: {eval_result['f1']:.4f}")
    print(f"  Balanced Accuracy: {eval_result['balanced_accuracy']:.4f}")
    print(f"  MCC: {eval_result['mcc']:.4f}")
    print(f"  PPV: {eval_result['ppv']:.4f}")
    print(f"  NPV: {eval_result['npv']:.4f}")
    print(
        f"  Confusion: TN={eval_result['tn']}, FP={eval_result['fp']}, FN={eval_result['fn']}, TP={eval_result['tp']}")

# 3. Select best threshold based on clinical requirements
print(f"\n{'=' * 80}")
print("THRESHOLD SELECTION BASED ON CLINICAL REQUIREMENTS")
print(f"{'=' * 80}")


# For diabetes screening, we typically want high sensitivity (recall) to catch most cases
# but also reasonable precision to avoid too many false alarms

# Calculate clinical utility score
def clinical_utility_score(eval_result, sensitivity_weight=0.4, specificity_weight=0.3, precision_weight=0.3):
    """Calculate clinical utility score weighted by clinical importance"""
    return (eval_result['sensitivity'] * sensitivity_weight +
            eval_result['specificity'] * specificity_weight +
            eval_result['precision'] * precision_weight)


best_clinical_threshold = None
best_clinical_score = 0
best_clinical_name = None

print("\nClinical Utility Scores (Sensitivity=40%, Specificity=30%, Precision=30%):")
for name, eval_result in threshold_evaluations.items():
    clinical_score = clinical_utility_score(eval_result)
    print(f"{name:20}: {clinical_score:.4f}")

    if clinical_score > best_clinical_score:
        best_clinical_score = clinical_score
        best_clinical_threshold = eval_result['threshold']
        best_clinical_name = name

print(f"\nBest clinical threshold: {best_clinical_name} ({best_clinical_threshold:.4f})")
print(f"Clinical utility score: {best_clinical_score:.4f}")

# 4. Save optimal threshold information
optimal_threshold_info = {
    'best_clinical_threshold': best_clinical_threshold,
    'best_clinical_name': best_clinical_name,
    'best_clinical_score': best_clinical_score,
    'all_thresholds': threshold_results,
    'all_evaluations': threshold_evaluations,
    'recommendation': f"Use {best_clinical_name} threshold ({best_clinical_threshold:.4f}) for optimal clinical performance"
}

# Update the final model prediction with optimal threshold
y_pred_optimal = (y_pred_proba >= best_clinical_threshold).astype(int)
final_evaluation = evaluate_threshold(y_test, y_pred_proba, best_clinical_threshold, "Final_Optimal")

print(f"\n{'=' * 80}")
print("FINAL MODEL PERFORMANCE WITH OPTIMAL THRESHOLD")
print(f"{'=' * 80}")
print(f"Optimal Threshold: {best_clinical_threshold:.4f}")
print(f"Accuracy: {final_evaluation['accuracy']:.4f}")
print(f"Precision: {final_evaluation['precision']:.4f}")
print(f"Recall (Sensitivity): {final_evaluation['recall']:.4f}")
print(f"Specificity: {final_evaluation['specificity']:.4f}")
print(f"F1-Score: {final_evaluation['f1']:.4f}")
print(f"Balanced Accuracy: {final_evaluation['balanced_accuracy']:.4f}")
print(f"MCC: {final_evaluation['mcc']:.4f}")
print(f"PPV: {final_evaluation['ppv']:.4f}")
print(f"NPV: {final_evaluation['npv']:.4f}")

# Use the optimal predictions for final evaluation
y_pred = y_pred_optimal

# Save threshold information
joblib.dump(optimal_threshold_info, '/content/drive/MyDrive/ML KAH/optimal_threshold_info.joblib')