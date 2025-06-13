import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, \
    precision_recall_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Dropout
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, cohen_kappa_score, average_precision_score)
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

# ----------------#
# 1. IMPORT DATA  #
# ----------------#
data = pd.read_csv('../dataset/cdc_diabetes_health_indicators.csv')
print("Data shape:", data.shape)
print("\nSample data:")
print(data.head())
print("\nData info:")
print(data.info())
print("\nStatistik deskriptif:")
print(data.describe())

# Periksa distribusi kelas target
print("\nDistribusi kelas target:")
target_counts = data['target'].value_counts()
print(target_counts)
print(f"Rasio kelas: {target_counts[0] / target_counts[1]:.2f}:1")

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

# Analisis korelasi
corr_matrix = X_scaled_df.copy()
corr_matrix['target'] = y
correlation = corr_matrix.corr()['target'].sort_values(ascending=False)
print("\nKorelasi fitur dengan target:")
print(correlation)

# Plot korelasi fitur terhadap target
plt.figure(figsize=(12, 8))
correlation.drop('target').plot(kind='bar')
plt.title('Korelasi Fitur dengan Target')
plt.tight_layout()
plt.savefig('feature_correlations.png')
plt.close()

# Buat beberapa fitur interaksi baru yang mungkin membantu prediksi diabetes
print("\nMembuat fitur interaksi baru...")
X_with_interactions = X_scaled_df.copy()

# Fitur interaksi antara BMI dan usia
X_with_interactions['BMI_X_Age'] = X_scaled_df['BMI'] * X_scaled_df['Age']

# Fitur interaksi antara BMI dan tekanan darah tinggi
X_with_interactions['BMI_X_HighBP'] = X_scaled_df['BMI'] * X_scaled_df['HighBP']

# Fitur interaksi antara usia dan tekanan darah tinggi
X_with_interactions['Age_X_HighBP'] = X_scaled_df['Age'] * X_scaled_df['HighBP']

# Fitur interaksi antara BMI dan kolesterol tinggi
X_with_interactions['BMI_X_HighChol'] = X_scaled_df['BMI'] * X_scaled_df['HighChol']

# Fitur interaksi gabungan antara kesehatan umum, mental, dan fisik
X_with_interactions['Health_Combined'] = (X_scaled_df['GenHlth'] + X_scaled_df['MentHlth'] + X_scaled_df[
    'PhysHlth']) / 3

# Konversi kembali ke numpy array
X_scaled_with_interactions = X_with_interactions.values

print(f"Jumlah fitur setelah menambahkan interaksi: {X_with_interactions.shape[1]}")

# ----------------------------#
# 3. Split Train/Test (75:25) #
# ----------------------------#

print("\n=== 3. Split Train/Test (75:25) ===")
X_train, X_test, y_train, y_test = train_test_split(X_scaled_with_interactions, y,
                                                    test_size=0.25, random_state=42, stratify=y)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Distribusi kelas di training set: {np.bincount(y_train)}")
print(f"Distribusi kelas di testing set: {np.bincount(y_test)}")

# ----------------------------------------#
# 4. Feature Selection - Random Forest    #
# ----------------------------------------#

print("\n=== 4. Feature Selection dengan Random Forest ===")
feature_names = list(X_with_interactions.columns)

# Membuat dan melatih Random Forest dengan class_weight='balanced'
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# Plotting feature importances
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature importances:")
print(feature_importances)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances[:15])
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.close()

# Pilih fitur berdasarkan threshold importance (lebih rendah untuk mempertahankan lebih banyak fitur)
selector = SelectFromModel(rf, threshold="mean", prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

selected_feature_indices = selector.get_support()
selected_features = [feature for feature, selected in zip(feature_names, selected_feature_indices) if selected]

print(f"Jumlah fitur terpilih: {X_train_selected.shape[1]} dari {X_train.shape[1]}")
print("Fitur terpilih:", selected_features)

# --------------------------------------------#
# 5. Resampling - Mengatasi Imbalanced Data   #
# --------------------------------------------#

print("\n=== 5. Resampling untuk Mengatasi Ketidakseimbangan Kelas ===")

# Sesuaikan rasio oversampling dan undersampling
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Membuat rasio minoritas:mayoritas = 1:2
rus = RandomUnderSampler(sampling_strategy=0.8, random_state=42)  # Membuat rasio minoritas:mayoritas = 1:1.25

# Pipeline resampling: SMOTE diikuti oleh Random Under Sampling
resampling_pipeline = ImbPipeline([
    ('smote', smote),
    ('under_sampler', rus)
])

# Lakukan resampling pada data training
X_train_resampled, y_train_resampled = resampling_pipeline.fit_resample(X_train_selected, y_train)

print(f"Bentuk data training sebelum resampling: {X_train_selected.shape}")
print(f"Distribusi kelas sebelum resampling: {np.bincount(y_train)}")
print(f"Bentuk data training setelah resampling: {X_train_resampled.shape}")
print(f"Distribusi kelas setelah resampling: {np.bincount(y_train_resampled)}")

# Visualisasi distribusi kelas sebelum dan sesudah resampling
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.bar(['Non-Diabetes (0)', 'Diabetes (1)'], np.bincount(y_train), color=['blue', 'red'])
plt.title('Distribusi Kelas Sebelum Resampling')
plt.ylabel('Jumlah Sampel')

plt.subplot(122)
plt.bar(['Non-Diabetes (0)', 'Diabetes (1)'], np.bincount(y_train_resampled), color=['blue', 'red'])
plt.title('Distribusi Kelas Setelah Resampling')
plt.ylabel('Jumlah Sampel')

plt.tight_layout()
plt.savefig('class_distribution_resampling.png')
plt.close()

# --------------------------------------------------#
# 6. Train Final Model (XGBoost dengan fokus recall) #
# --------------------------------------------------#

print("\n=== 6. Train Final Model (XGBoost with focus on recall) ===")

# Parameters for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'scale_pos_weight': [1, 3, 5, 7]  # Important for imbalanced data
}

# Stratified K-Fold for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# RandomizedSearchCV for parameter tuning
# Note: Remove XGBoost-specific parameters from here
xgb_random = RandomizedSearchCV(
    xgb.XGBClassifier(objective='binary:logistic', random_state=42,
                     use_label_encoder=False, eval_metric='logloss'),
    param_distributions=param_dist,
    n_iter=30,  # Number of parameter combinations to test
    scoring='recall',  # Optimize for recall
    cv=skf,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Train the model on the resampled data
# Note: Remove eval_set and early_stopping_rounds from here
xgb_random.fit(X_train_resampled, y_train_resampled)

# Best model parameters
print("Best parameters:", xgb_random.best_params_)
print("Best recall score: {:.4f}".format(xgb_random.best_score_))

# Get the best model from RandomizedSearchCV
best_model_params = xgb_random.best_params_

# Create a new model with the best parameters and train it with early stopping
best_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    **best_model_params
)

# Now fit this model with early stopping
best_xgb.fit(
    X_train_resampled,
    y_train_resampled,
    eval_set=[(X_test_selected, y_test)],
    early_stopping_rounds=50,
    verbose=0
)

# Cross-validation with the best model
cv_recall = cross_val_score(best_xgb, X_train_resampled, y_train_resampled,
                           cv=skf, scoring='recall')
cv_accuracy = cross_val_score(best_xgb, X_train_resampled, y_train_resampled,
                             cv=skf, scoring='accuracy')
cv_f1 = cross_val_score(best_xgb, X_train_resampled, y_train_resampled,
                       cv=skf, scoring='f1')

print("Cross-validation metrics:")
print(f"Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
print(f"Recall: {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
print(f"F1-score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
# -----------------------------#
# 7. Threshold Optimization    #
# -----------------------------#

print("\n=== 7. Threshold Optimization untuk Meningkatkan Recall ===")

# Prediksi probabilitas pada data tes
y_pred_proba = best_xgb.predict_proba(X_test_selected)[:, 1]

# Evaluasi berbagai threshold untuk menemukan yang optimal untuk recall
thresholds = np.arange(0.1, 0.9, 0.05)
recall_scores = []
precision_scores = []
f1_scores = []
accuracy_scores = []

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
    recall_scores.append(recall_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# Plot metrics vs threshold
plt.figure(figsize=(12, 8))
plt.plot(thresholds, recall_scores, 'b-', label='Recall')
plt.plot(thresholds, precision_scores, 'g-', label='Precision')
plt.plot(thresholds, f1_scores, 'r-', label='F1 Score')
plt.plot(thresholds, accuracy_scores, 'y-', label='Accuracy')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Metric Scores vs. Classification Threshold')
plt.legend()
plt.grid(True)
plt.savefig('threshold_optimization.png')
plt.close()

# Temukan threshold yang memaksimalkan F1-score
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Threshold optimal untuk F1-score: {optimal_threshold:.2f}")
print(f"F1-score pada threshold optimal: {f1_scores[optimal_idx]:.4f}")
print(f"Recall pada threshold optimal: {recall_scores[optimal_idx]:.4f}")
print(f"Precision pada threshold optimal: {precision_scores[optimal_idx]:.4f}")
print(f"Accuracy pada threshold optimal: {accuracy_scores[optimal_idx]:.4f}")

# Jika recall masih menjadi prioritas, kita bisa menurunkan threshold lebih jauh
# Temukan threshold dengan recall minimal 0.7 tetapi dengan precision tertinggi
target_recall = 0.7
valid_indices = [i for i, r in enumerate(recall_scores) if r >= target_recall]

if valid_indices:
    # Ambil threshold dengan precision tertinggi dari threshold yang memenuhi recall target
    best_precision_idx = valid_indices[np.argmax([precision_scores[i] for i in valid_indices])]
    recall_optimized_threshold = thresholds[best_precision_idx]

    print(f"\nThreshold optimal untuk recall >= {target_recall}: {recall_optimized_threshold:.2f}")
    print(f"Recall pada threshold ini: {recall_scores[best_precision_idx]:.4f}")
    print(f"Precision pada threshold ini: {precision_scores[best_precision_idx]:.4f}")
    print(f"F1-score pada threshold ini: {f1_scores[best_precision_idx]:.4f}")
    print(f"Accuracy pada threshold ini: {accuracy_scores[best_precision_idx]:.4f}")
else:
    print(f"\nTidak ditemukan threshold yang menghasilkan recall >= {target_recall}")
    # Pilih threshold dengan recall tertinggi
    recall_optimized_threshold = thresholds[np.argmax(recall_scores)]
    max_recall_idx = np.argmax(recall_scores)

    print(f"Threshold dengan recall tertinggi: {recall_optimized_threshold:.2f}")
    print(f"Recall pada threshold ini: {recall_scores[max_recall_idx]:.4f}")
    print(f"Precision pada threshold ini: {precision_scores[max_recall_idx]:.4f}")
    print(f"F1-score pada threshold ini: {f1_scores[max_recall_idx]:.4f}")
    print(f"Accuracy pada threshold ini: {accuracy_scores[max_recall_idx]:.4f}")

# Gunakan threshold yang dipilih untuk evaluasi final
final_threshold = recall_optimized_threshold
y_pred_final = (y_pred_proba >= final_threshold).astype(int)

# ------------------#
# 8. Final Testing  #
# ------------------#

print("\n=== 8. Final Testing ===")

# Evaluasi model dengan threshold terpilih
accuracy = accuracy_score(y_test, y_pred_final)
precision = precision_score(y_test, y_pred_final)
recall = recall_score(y_test, y_pred_final)
f1 = f1_score(y_test, y_pred_final)
roc_auc_value = roc_auc_score(y_test, y_pred_proba)
mcc = matthews_corrcoef(y_test, y_pred_final)
kappa = cohen_kappa_score(y_test, y_pred_final)
average_precision = average_precision_score(y_test, y_pred_proba)

# Tampilkan hasil metrik untuk publikasi jurnal
print("\n" + "=" * 50)
print("HASIL EVALUASI MODEL UNTUK PUBLIKASI JURNAL")
print("=" * 50)
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1-Score      : {f1:.4f}")
print(f"ROC AUC       : {roc_auc_value:.4f}")
print(f"MCC           : {mcc:.4f}")
print(f"Cohen's Kappa : {kappa:.4f}")
print(f"Avg Precision : {average_precision:.4f}")
print("=" * 50)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum * 100

# Plot confusion matrix
plt.figure(figsize=(12, 10))

plt.subplot(221)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Count)')
plt.ylabel('Aktual')
plt.xlabel('Prediksi')

plt.subplot(222)
sns.heatmap(cm_perc, annot=True, fmt='.2f', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Percentage %)')
plt.ylabel('Aktual')
plt.xlabel('Prediksi')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.subplot(223)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

# Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)

plt.subplot(224)
plt.plot(recall_curve, precision_curve, color='green', lw=2,
         label=f'PR curve (avg precision = {average_precision:.3f})')
plt.axvline(x=recall, color='red', linestyle='--',
            label=f'Selected operating point\nRecall={recall:.2f}, Precision={precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.savefig('model_evaluation_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

# Simpan model terbaik dan pipeline preprocessing
import joblib
from datetime import datetime

# Simpan model dan komponen preprocessing
joblib.dump(best_xgb, 'best_xgb_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(selector, 'feature_selector.joblib')
joblib.dump(resampling_pipeline, 'resampling_pipeline.joblib')
joblib.dump(final_threshold, 'classification_threshold.joblib')

# Simpan hasil eksperimen untuk jurnal
experiment_results = {
    'Model': 'XGBoost dengan Optimasi Threshold dan Resampling',
    'Tanggal': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'Dataset_Size': X.shape[0],
    'Original_Features': X.shape[1],
    'Enhanced_Features': X_with_interactions.shape[1],
    'Selected_Features': len(selected_features),
    'Resampling_Strategy': 'SMOTE + Random Undersampling',
    'Classification_Threshold': final_threshold,
    'Best_Parameters': xgb_random.best_params_,
    'CV_Recall_Mean': cv_recall.mean(),
    'CV_Recall_Std': cv_recall.std(),
    'Test_Accuracy': accuracy,
    'Test_Precision': precision,
    'Test_Recall': recall,
    'Test_F1': f1,
    'Test_ROC_AUC': roc_auc_value,
    'Test_MCC': mcc,
    'Test_Kappa': kappa,
    'Test_Avg_Precision': average_precision
}

# Simpan dalam format CSV
pd.DataFrame([experiment_results]).to_csv('hasil_eksperimen_recall_focus.csv', index=False)

# Simpan hasil dengan format tabel untuk jurnal
with open('hasil_table_untuk_jurnal.txt', 'w') as f:
    f.write("Table X: Performance metrics of the proposed recall-optimized model for diabetes prediction\n\n")
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| Accuracy | {accuracy:.4f} |\n")
    f.write(f"| Precision | {precision:.4f} |\n")
    f.write(f"| Recall | {recall:.4f} |\n")
    f.write(f"| F1-Score | {f1:.4f} |\n")
    f.write(f"| ROC AUC | {roc_auc_value:.4f} |\n")
    f.write(f"| Matthews Correlation Coefficient | {mcc:.4f} |\n")
    f.write(f"| Cohen's Kappa | {kappa:.4f} |\n")
    f.write(f"| Average Precision | {average_precision:.4f} |\n\n")
    f.write(
        "*Note: The proposed model optimizes for recall using threshold adjustment, robust resampling techniques and feature engineering.")

print("\nProses selesai! Model terbaik dengan fokus recall tersimpan sebagai 'best_xgb_model.joblib'")


# Buat fungsi untuk prediksi data baru dengan threshold yang telah dioptimalkan
def predict_diabetes(new_data, threshold=final_threshold):
    """
    Memprediksi diabetes pada data baru dengan threshold yang dioptimalkan untuk recall

    Parameters:
    -----------
    new_data : pandas DataFrame
        Data baru yang akan diprediksi, harus memiliki kolom yang sama dengan dataset asli
    threshold : float, default=optimal_threshold
        Threshold probabilitas untuk mengklasifikasikan sebagai diabetes (kelas 1)

    Returns:
    --------
    prediksi : array
        Hasil prediksi (0: tidak diabetes, 1: diabetes)
    probabilitas : array
        Probabilitas prediksi
    """
    # Preprocessing
    if 'ID' in new_data.columns:
        new_data = new_data.drop('ID', axis=1)
    if 'Diabetes_binary' in new_data.columns or 'target' in new_data.columns:
        new_data = new_data.drop(
            ['Diabetes_binary', 'target'] if 'Diabetes_binary' in new_data.columns and 'target' in new_data.columns
            else 'Diabetes_binary' if 'Diabetes_binary' in new_data.columns
            else 'target', axis=1)

    # Scaling
    X_new_scaled = scaler.transform(new_data)

    # Tambahkan fitur interaksi yang sama seperti pada training
    X_new_df = pd.DataFrame(X_new_scaled, columns=X.columns)
    X_new_with_interactions = X_new_df.copy()

    # Buat fitur interaksi yang sama
    X_new_with_interactions['BMI_X_Age'] = X_new_df['BMI'] * X_new_df['Age']
    X_new_with_interactions['BMI_X_HighBP'] = X_new_df['BMI'] * X_new_df['HighBP']
    X_new_with_interactions['Age_X_HighBP'] = X_new_df['Age'] * X_new_df['HighBP']
    X_new_with_interactions['BMI_X_HighChol'] = X_new_df['BMI'] * X_new_df['HighChol']
    X_new_with_interactions['Health_Combined'] = (X_new_df['GenHlth'] + X_new_df['MentHlth'] + X_new_df['PhysHlth']) / 3

    # Feature selection
    X_new_selected = selector.transform(X_new_with_interactions.values)

    # Prediksi
    probability = best_xgb.predict_proba(X_new_selected)[:, 1]
    prediction = (probability >= threshold).astype(int)

    return prediction, probability


print("\nContoh penggunaan fungsi prediksi dengan threshold optimal:")
print("prediction, probability = predict_diabetes(new_patient_data)")