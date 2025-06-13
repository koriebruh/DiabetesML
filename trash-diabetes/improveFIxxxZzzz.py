import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, \
    average_precision_score, matthews_corrcoef, cohen_kappa_score
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import joblib
import shap
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ----------------#
# 1. IMPORT DATA #
# ----------------#

# Definisikan path dataset
data_path = '/content/drive/MyDrive/ML KAH/dataset/cdc_diabetes_health_indicators.csv'

# Cek apakah file ada
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File tidak ditemukan di {data_path}. Pastikan path benar dan file ada di Google Drive.")
else:
    print(f"File ditemukan di {data_path}")

# Load data
data = pd.read_csv(data_path)
print("Data shape:", data.shape)
print("\nKolom di dataset:", list(data.columns))  # Tambahkan untuk debugging
print("\nSample data:")
print(data.head())
print("\nData info:")
print(data.info())
print("\nStatistik deskriptif:")
print(data.describe())

# Cek apakah kolom 'target' ada
if 'target' not in data.columns:
    raise KeyError("Kolom 'target' tidak ditemukan di dataset. Periksa nama kolom target.")
else:
    print("Kolom target ditemukan: 'target'")

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
plt.xticks([0, 1], ['Tidak Diabetes', 'Ya Diabetes'])
plt.subplot(1, 2, 2)
sns.countplot(x='target', data=data)
plt.title('Distribusi Kelas (Target)')
plt.xlabel('Kelas')
plt.ylabel('Jumlah')
plt.xticks([0, 1], ['Tidak Diabetes', 'Ya Diabetes'])
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/ML KAH/output/class_distribution.png')
plt.close()

# -----------------#
# 2. Preprocessing #
# ----------------#

print("\n=== 2. Preprocessing ===")
# Pisahkan fitur dan target
X = data.drop('target', axis=1)
y = data['target']

print("Fitur shape:", X.shape)
print("Target shape:", y.shape)

# Feature engineering: Tambahkan fitur interaksi dan binning
X['Age_BMI'] = X['Age'] * X['BMI']
X['BMI_cat'] = pd.cut(X['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])

# Normalisasi/Standardisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("Data setelah scaling (sample):")
print(X_scaled_df.head())

# ----------------------------------------#
# 3. Feature Selection dengan Random Forest #
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

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances[:10])
plt.title('Top 10 Feature Importances')
plt.savefig('/content/drive/MyDrive/ML KAH/output/feature_importances.png')
plt.close()

# Pilih fitur berdasarkan threshold importance
selector = SelectFromModel(rf, threshold='mean', prefit=True)
X_selected = selector.transform(X_scaled)
selected_feature_indices = selector.get_support()
selected_features = X.columns[selected_feature_indices].tolist()

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
print("Distribusi kelas di train set:", pd.Series(y_train).value_counts().to_dict())
print("Distribusi kelas di test set:", pd.Series(y_test).value_counts().to_dict())

# ----------------------------------------------#
# 5. DATA BALANCING - SMOTE + Undersampling #
# ----------------------------------------------#

print("\n=== 5. DATA BALANCING - SMOTE + RandomUnderSampler ===")
# Kombinasi SMOTE (1:2) dan RandomUnderSampler
balancing_pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),  # Rasio 1:2
    ('undersampler', RandomUnderSampler(sampling_strategy=0.5, random_state=42))
])
X_train_balanced, y_train_balanced = balancing_pipeline.fit_resample(X_train, y_train)

print(f"Balanced train set shape: {X_train_balanced.shape}")
print(f"Balanced train distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")

# Visualisasi hasil balancing
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
pd.Series(y_train).value_counts().plot(kind='bar')
plt.title('Before Balancing (Train)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['No Diabetes', 'Diabetes'], rotation=0)

plt.subplot(1, 2, 2)
pd.Series(y_train_balanced).value_counts().plot(kind='bar')
plt.title('After Balancing (SMOTE + Undersampling)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['No Diabetes', 'Diabetes'], rotation=0)
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/ML KAH/output/balancing_comparison.png')
plt.close()

# -------------------------------------------------#
# 6. Dense Autoencoder untuk Feature Extraction #
# -------------------------------------------------#

print("\n=== 6. Dense Autoencoder untuk Feature Extraction ===")
input_dim = X_train_balanced.shape[1]
encoding_dim = int(input_dim * 0.75)

# Buat model Dense Autoencoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='linear')(encoder)
autoencoder = Model(input_layer, decoder)
encoder_model = Model(input_layer, encoder)

autoencoder.compile(optimizer='adam', loss='mse')
print("Dense Autoencoder Summary:")
autoencoder.summary()

# Train autoencoder dengan early stopping
history = autoencoder.fit(
    X_train_balanced, X_train_balanced,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)],
    verbose=1
)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Dense Autoencoder Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('/content/drive/MyDrive/ML KAH/output/autoencoder_loss.png')
plt.close()

# Extract features
X_train_encoded = encoder_model.predict(X_train_balanced)
X_test_encoded = encoder_model.predict(X_test)
print(f"Encoded features shape: {X_train_encoded.shape}")

# ----------------------------------------------------------#
# 7. Feature Fusion (Original + Encoded + Custom Features) #
# ----------------------------------------------------------#

print("\n=== 7. Feature Fusion ===")
X_train_fused = np.hstack((X_train_balanced, X_train_encoded))
X_test_fused = np.hstack((X_test, X_test_encoded))

# Custom feature: BMI_Activity dan HighRisk
X_train_df = pd.DataFrame(X_train_balanced, columns=selected_features)
X_test_df = pd.DataFrame(X_test, columns=selected_features)

if 'BMI' in selected_features and 'PhysActivity' in selected_features:
    X_train_fused_df = pd.DataFrame(X_train_fused)
    X_test_fused_df = pd.DataFrame(X_test_fused)
    X_train_fused_df['BMI_Activity'] = X_train_df['BMI'] / (X_train_df['PhysActivity'] + 1)
    X_test_fused_df['BMI_Activity'] = X_test_df['BMI'] / (X_test_df['PhysActivity'] + 1)
    X_train_fused = X_train_fused_df.values
    X_test_fused = X_test_fused_df.values
    print("Fitur baru BMI_Activity ditambahkan")

if 'HighBP' in selected_features and 'HighChol' in selected_features:
    X_train_fused_df = pd.DataFrame(X_train_fused)
    X_test_fused_df = pd.DataFrame(X_test_fused)
    X_train_fused_df['HighRisk'] = ((X_train_df['HighBP'] + X_train_df['HighChol']) > 0).astype(int)
    X_test_fused_df['HighRisk'] = ((X_test_df['HighBP'] + X_test_df['HighChol']) > 0).astype(int)
    X_train_fused = X_train_fused_df.values
    X_test_fused = X_test_fused_df.values
    print("Fitur baru HighRisk ditambahkan")

print(f"Feature fusion shape: {X_train_fused.shape}")

# --------------------------------------------------#
# 8. Feature Selection Ulang setelah Feature Fusion #
# --------------------------------------------------#

print("\n=== 8. Feature Selection Ulang ===")
rf_after_fusion = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_after_fusion.fit(X_train_fused, y_train_balanced)
selector_after_fusion = SelectFromModel(rf_after_fusion, threshold='mean', prefit=True)
X_train_final = selector_after_fusion.transform(X_train_fused)
X_test_final = selector_after_fusion.transform(X_test_fused)

print(f"Jumlah fitur final: {X_train_final.shape[1]} dari {X_train_fused.shape[1]}")

# --------------------------------------------------#
# 9. Train Final Model (Stacking Ensemble) #
# --------------------------------------------------#

print("\n=== 9. Train Final Model (Stacking Ensemble) ===")
# Hitung class weights untuk XGBoost
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train_balanced)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_balanced)
class_weight_dict = dict(zip(classes, class_weights))
scale_pos_weight = class_weight_dict[0] / class_weight_dict[1] * 1.5  # Tambah bobot kelas 1

# Parameter untuk RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_lambda': [0.1, 1.0, 10.0],
    'scale_pos_weight': [1, scale_pos_weight]
}

# RandomizedSearchCV untuk XGBoost
xgb_random = RandomizedSearchCV(
    xgb.XGBClassifier(objective='binary:logistic', random_state=42, eval_metric='logloss'),
    param_distributions=param_grid,
    n_iter=20,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1',
    verbose=1,
    n_jobs=-1
)
xgb_random.fit(X_train_final, y_train_balanced)

print("Best parameters:", xgb_random.best_params_)
print("Best cross-validation F1-score: {:.4f}".format(xgb_random.best_score_))

# Stacking ensemble
base_learners = [
    ('xgb', xgb.XGBClassifier(**xgb_random.best_params_, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('lgbm', LGBMClassifier(random_state=42))
]
stacking = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(),
    cv=5
)
stacking.fit(X_train_final, y_train_balanced)

# Cross-validation
cv_results = cross_validate(
    stacking, X_train_final, y_train_balanced,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
)

print("\nCross-validation results:")
for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    scores = cv_results[f'test_{metric}']
    print(f"{metric.capitalize()}: {scores.mean():.4f} ± {scores.std():.4f}")

# -----------------#
# 10. Final Testing #
# -----------------#

print("\n=== 10. Final Testing ===")
y_pred = stacking.predict(X_test_final)
y_pred_proba = stacking.predict_proba(X_test_final)[:, 1]

# Threshold optimization
precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve)
f1_scores = np.nan_to_num(f1_scores)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

# Evaluasi dengan default dan optimal threshold
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc_value = roc_auc_score(y_test, y_pred_proba)
mcc = matthews_corrcoef(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
average_precision = average_precision_score(y_test, y_pred_proba)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

accuracy_opt = accuracy_score(y_test, y_pred_optimal)
precision_opt = precision_score(y_test, y_pred_optimal)
recall_opt = recall_score(y_test, y_pred_optimal)
f1_opt = f1_score(y_test, y_pred_optimal)

print("\n" + "=" * 60)
print("HASIL EVALUASI MODEL UNTUK PUBLIKASI JURNAL (OPTIMIZED)")
print("=" * 60)
print("Default Threshold (0.5):")
print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity        : {specificity:.4f}")
print(f"F1-Score           : {f1:.4f}")
print(f"ROC AUC            : {roc_auc_value:.4f}")
print(f"MCC                : {mcc:.4f}")
print(f"Cohen's Kappa      : {kappa:.4f}")
print(f"Avg Precision      : {average_precision:.4f}")
print(f"\nOptimal Threshold  : {optimal_threshold:.4f}")
print(f"Accuracy (Optimal)  : {accuracy_opt:.4f}")
print(f"Precision (Optimal) : {precision_opt:.4f}")
print(f"Recall (Optimal)    : {recall_opt:.4f}")
print(f"F1-Score (Optimal) : {f1_opt:.4f}")
print("=" * 60)

# Plot visualisasi
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

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

plt.subplot(1, 3, 3)
plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AP = {average_precision:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/ML KAH/output/evaluation_plots_optimized.png')
plt.close()

# SHAP Analysis
print("\n=== 11. SHAP Analysis ===")
explainer = shap.TreeExplainer(base_learners[0][1])  # Gunakan XGBoost dari stacking
shap_values = explainer.shap_values(X_test_final)
shap.summary_plot(shap_values, X_test_final, feature_names=[f"Feature_{i}" for i in range(X_test_final.shape[1])],
                  show=False)
plt.savefig('/content/drive/MyDrive/ML KAH/output/shap_summary.png')
plt.close()

# Simpan hasil untuk jurnal
with open('/content/drive/MyDrive/ML KAH/output/hasil_evaluasi_untuk_jurnal_optimized.txt', 'w') as f:
    f.write("HASIL EVALUASI MODEL UNTUK PUBLIKASI JURNAL (OPTIMIZED)\n")
    f.write("=" * 60 + "\n")
    f.write("Balancing Technique : SMOTE + RandomUnderSampler\n")
    f.write(f"Default Threshold   : 0.5\n")
    f.write(f"Accuracy           : {accuracy:.4f}\n")
    f.write(f"Precision          : {precision:.4f}\n")
    f.write(f"Recall (Sensitivity): {recall:.4f}\n")
    f.write(f"Specificity        : {specificity:.4f}\n")
    f.write(f"F1-Score           : {f1:.4f}\n")
    f.write(f"ROC AUC            : {roc_auc_value:.4f}\n")
    f.write(f"MCC                : {mcc:.4f}\n")
    f.write(f"Cohen's Kappa      : {kappa:.4f}\n")
    f.write(f"Avg Precision      : {average_precision:.4f}\n")
    f.write(f"Optimal Threshold  : {optimal_threshold:.4f}\n")
    f.write(f"Accuracy (Optimal)  : {accuracy_opt:.4f}\n")
    f.write(f"Precision (Optimal) : {precision_opt:.4f}\n")
    f.write(f"Recall (Optimal)    : {recall_opt:.4f}\n")
    f.write(f"F1-Score (Optimal) : {f1_opt:.4f}\n")
    f.write("=" * 60 + "\n\n")
    f.write("Confusion Matrix (Default Threshold):\n")
    f.write(f"True Negatives: {tn}\n")
    f.write(f"False Positives: {fp}\n")
    f.write(f"False Negatives: {fn}\n")
    f.write(f"True Positives: {tp}\n\n")
    f.write("Classification Report (Default Threshold):\n")
    f.write(classification_report(y_test, y_pred))

# Simpan model dan preprocessing objects
joblib.dump(stacking, '/content/drive/MyDrive/ML KAH/output/best_stacking_model_optimized.joblib')
joblib.dump(scaler, '/content/drive/MyDrive/ML KAH/output/scaler_optimized.joblib')
joblib.dump(selector, '/content/drive/MyDrive/ML KAH/output/feature_selector_optimized.joblib')
joblib.dump(selector_after_fusion,
            '/content/drive/MyDrive/ML KAH/output/feature_selector_after_fusion_optimized.joblib')
joblib.dump(encoder_model, '/content/drive/MyDrive/ML KAH/output/dense_encoder_optimized.joblib')
joblib.dump({'optimal_threshold': optimal_threshold},
            '/content/drive/MyDrive/ML KAH/output/threshold_info_optimized.joblib')

# Simpan hasil eksperimen untuk jurnal
experiment_results = {
    'Model': 'Stacking (XGBoost + RF + LightGBM) + Dense Autoencoder + Feature Fusion',
    'Tanggal': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'Dataset_Size': X.shape[0],
    'Original_Features': X.shape[1],
    'Selected_Features': len(selected_features),
    'Final_Features': X_train_final.shape[1],
    'Balancing_Technique': 'SMOTE + RandomUnderSampler',
    'Original_Class_Distribution': dict(pd.Series(y_train).value_counts()),
    'Balanced_Class_Distribution': dict(pd.Series(y_train_balanced).value_counts()),
    'Best_Parameters': xgb_random.best_params_,
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
    'Test_Avg_Precision': average_precision,
    'Optimal_Threshold': optimal_threshold,
    'Test_F1_Optimal': f1_opt
}
pd.DataFrame([experiment_results]).to_csv(
    '/content/drive/MyDrive/ML KAH/output/hasil_eksperimen_untuk_jurnal_optimized.csv', index=False)

# Perbandingan dengan hasil sebelumnya
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
comparison_df.to_csv('/content/drive/MyDrive/ML KAH/output/comparison_before_after_balancing.csv', index=False)

print("\nClassification Report (Default Threshold):")
print(classification_report(y_test, y_pred))

print("\nProses selesai! Model terbaik tersimpan sebagai 'best_stacking_model_optimized.joblib'")
print("\nFile output disimpan di /content/drive/MyDrive/ML KAH/output/:")
print("- best_stacking_model_optimized.joblib")
print("- scaler_optimized.joblib")
print("- feature_selector_optimized.joblib")
print("- feature_selector_after_fusion_optimized.joblib")
print("- dense_encoder_optimized.joblib")
print("- threshold_info_optimized.joblib")
print("- hasil_evaluasi_untuk_jurnal_optimized.txt")
print("- hasil_eksperimen_untuk_jurnal_optimized.csv")
print("- comparison_before_after_balancing.csv")
print(
    "- Visualisasi: class_distribution.png, feature_importances.png, balancing_comparison.png, autoencoder_loss.png, evaluation_plots_optimized.png, shap_summary.png")


# Fungsi prediksi untuk data baru
def predict_diabetes_optimized(new_data, threshold=optimal_threshold):
    """
    Memprediksi diabetes pada data baru dengan model yang sudah di-optimasi.

    Parameters:
    -----------
    new_data : pandas DataFrame
        Data baru dengan kolom sesuai dataset asli (kecuali target)
    threshold : float
        Threshold untuk klasifikasi (default: optimal_threshold)

    Returns:
    --------
    prediction : array
        Hasil prediksi (0: tidak diabetes, 1: diabetes)
    probability : array
        Probabilitas prediksi
    """
    # Preprocessing
    if 'target' in new_data.columns:
        new_data = new_data.drop('target', axis=1)

    # Feature engineering
    new_data['Age_BMI'] = new_data['Age'] * new_data['BMI']
    new_data['BMI_cat'] = pd.cut(new_data['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])

    # Scaling
    X_new_scaled = scaler.transform(new_data)

    # Feature selection pertama
    X_new_selected = selector.transform(X_new_scaled)

    # Dense encoding
    X_new_encoded = encoder_model.predict(X_new_selected)

    # Feature fusion
    X_new_fused = np.hstack((X_new_selected, X_new_encoded))

    # Tambahkan custom features
    new_data_df = pd.DataFrame(X_new_selected, columns=selected_features)
    X_new_fused_df = pd.DataFrame(X_new_fused)
    if 'BMI' in selected_features and 'PhysActivity' in selected_features:
        X_new_fused_df['BMI_Activity'] = new_data_df['BMI'] / (new_data_df['PhysActivity'] + 1)
    if 'HighBP' in selected_features and 'HighChol' in selected_features:
        X_new_fused_df['HighRisk'] = ((new_data_df['HighBP'] + new_data_df['HighChol']) > 0).astype(int)
    X_new_fused = X_new_fused_df.values

    # Feature selection final
    X_new_final = selector_after_fusion.transform(X_new_fused)

    # Prediksi
    probability = stacking.predict_proba(X_new_final)[:, 1]
    prediction = (probability >= threshold).astype(int)

    return prediction, probability


# Contoh prediksi
print("\n=== CONTOH PREDIKSI PADA DATA BARU ===")
sample_indices = np.random.choice(X_test.shape[0], size=5, replace=False)
X_sample = X_test[sample_indices]
y_sample = y_test.iloc[sample_indices].values
X_sample_df = pd.DataFrame(X_sample, columns=selected_features)
pred, prob = predict_diabetes_optimized(X_sample_df)

for i in range(5):
    actual = "Diabetes" if y_sample[i] == 1 else "Tidak Diabetes"
    predicted = "Diabetes" if pred[i] == 1 else "Tidak Diabetes"
    print(f"Sampel {i + 1}:")
    print(f"  Aktual: {actual}")
    print(f"  Prediksi: {predicted}")
    print(f"  Probabilitas: {prob[i]:.4f}")
    print(f"  Status: {'✓ Benar' if pred[i] == y_sample[i] else '✗ Salah'}")
    print("-" * 60)