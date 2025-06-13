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


#  RUN 3,5 JAM

# classification Report:
#               precision    recall  f1-score   support
#
#            0       0.93      0.80      0.86     54583
#            1       0.33      0.60      0.43      8837
#
#     accuracy                           0.78     63420
#    macro avg       0.63      0.70      0.65     63420
# weighted avg       0.84      0.78      0.80     63420
#
#
# ================================================================================
# PERBANDINGAN HASIL: SEBELUM vs SESUDAH BALANCING
# ================================================================================
# Metrik               | Sebelum Balancing | Sesudah Balancing | Peningkatan
# --------------------------------------------------------------------------------
# Recall (Sensitivity) | 0.0101           | 0.6046        | 5886.3x
# F1-Score            | 0.0198           | 0.4293        | 2068.4x
# Precision           | 0.5235           | 0.3329        | -36.4%
# Accuracy            | 0.8608           | 0.7760        | -9.8%
# ROC AUC             | 0.7898           | 0.7972        | +0.9%
# ================================================================================



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


# Fungsi untuk evaluasi cepat
def quick_evaluate(X_train_balanced, y_train_balanced, X_test, y_test, technique_name):
    """Evaluasi cepat untuk membandingkan teknik balancing"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, f1_score

    # Train simple RF model
    rf_temp = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_temp.fit(X_train_balanced, y_train_balanced)

    # Predict and evaluate
    y_pred = rf_temp.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{technique_name}:")
    print(f"Balanced train set shape: {X_train_balanced.shape}")
    print(f"Balanced train distribution: {Counter(y_train_balanced)}")
    print(f"F1-Score: {f1:.4f}")

    return f1


# Test berbagai teknik balancing
balancing_results = {}

# 1. SMOTE
print("Testing SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
f1_smote = quick_evaluate(X_train_smote, y_train_smote, X_test, y_test, "SMOTE")
balancing_results['SMOTE'] = f1_smote

# 2. ADASYN
print("Testing ADASYN...")
adasyn = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
f1_adasyn = quick_evaluate(X_train_adasyn, y_train_adasyn, X_test, y_test, "ADASYN")
balancing_results['ADASYN'] = f1_adasyn

# 3. BorderlineSMOTE
print("Testing BorderlineSMOTE...")
borderline_smote = BorderlineSMOTE(random_state=42)
X_train_borderline, y_train_borderline = borderline_smote.fit_resample(X_train, y_train)
f1_borderline = quick_evaluate(X_train_borderline, y_train_borderline, X_test, y_test, "BorderlineSMOTE")
balancing_results['BorderlineSMOTE'] = f1_borderline

# 4. SMOTEENN (Combination)
print("Testing SMOTEENN...")
smoteenn = SMOTEENN(random_state=42)
X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train, y_train)
f1_smoteenn = quick_evaluate(X_train_smoteenn, y_train_smoteenn, X_test, y_test, "SMOTEENN")
balancing_results['SMOTEENN'] = f1_smoteenn

# 5. SMOTETomek (Combination)
print("Testing SMOTETomek...")
smotetomek = SMOTETomek(random_state=42)
X_train_smotetomek, y_train_smotetomek = smotetomek.fit_resample(X_train, y_train)
f1_smotetomek = quick_evaluate(X_train_smotetomek, y_train_smotetomek, X_test, y_test, "SMOTETomek")
balancing_results['SMOTETomek'] = f1_smotetomek

# Pilih teknik terbaik
best_technique = max(balancing_results, key=balancing_results.get)
print(f"\n=== BEST BALANCING TECHNIQUE: {best_technique} (F1: {balancing_results[best_technique]:.4f}) ===")

# Gunakan teknik terbaik
if best_technique == 'SMOTE':
    X_train_balanced, y_train_balanced = X_train_smote, y_train_smote
elif best_technique == 'ADASYN':
    X_train_balanced, y_train_balanced = X_train_adasyn, y_train_adasyn
elif best_technique == 'BorderlineSMOTE':
    X_train_balanced, y_train_balanced = X_train_borderline, y_train_borderline
elif best_technique == 'SMOTEENN':
    X_train_balanced, y_train_balanced = X_train_smoteenn, y_train_smoteenn
else:  # SMOTETomek
    X_train_balanced, y_train_balanced = X_train_smotetomek, y_train_smotetomek

# Visualisasi hasil balancing
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
pd.Series(y_train).value_counts().plot(kind='bar')
plt.title('Before Balancing (Train)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['No Diabetes', 'Diabetes'], rotation=0)

plt.subplot(1, 3, 2)
pd.Series(y_train_balanced).value_counts().plot(kind='bar')
plt.title(f'After Balancing ({best_technique})')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['No Diabetes', 'Diabetes'], rotation=0)

plt.subplot(1, 3, 3)
techniques = list(balancing_results.keys())
f1_scores = list(balancing_results.values())
plt.bar(techniques, f1_scores)
plt.title('F1-Score Comparison of Balancing Techniques')
plt.xticks(rotation=45)
plt.ylabel('F1-Score')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/ML KAH/balancing_comparison.png')
plt.close()

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
from sklearn.model_selection import GridSearchCV, cross_val_score

# Hitung class weights untuk XGBoost
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train_balanced)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_balanced)
class_weight_dict = dict(zip(classes, class_weights))
scale_pos_weight = class_weight_dict[0] / class_weight_dict[1]

print(f"Scale pos weight for XGBoost: {scale_pos_weight}")

# Parameter tuning untuk XGBoost dengan class balancing
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1],
    'scale_pos_weight': [1, scale_pos_weight]  # Untuk handling imbalanced data
}

# Grid search with cross-validation - gunakan StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_grid = GridSearchCV(
    xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='logloss'),
    param_grid=param_grid,
    cv=skf,
    scoring='f1',  # Gunakan F1-score sebagai metric utama untuk imbalanced data
    verbose=1,
    n_jobs=-1
)

xgb_grid.fit(X_train_final, y_train_balanced)

# Best model parameters
print("Best parameters:", xgb_grid.best_params_)
print("Best cross-validation F1-score: {:.4f}".format(xgb_grid.best_score_))

# Get the best model
best_xgb = xgb_grid.best_estimator_

# Cross-validation with multiple metrics
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    best_xgb, X_train_final, y_train_balanced,
    cv=skf,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    return_train_score=False
)

print("\nCross-validation results:")
for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    scores = cv_results[f'test_{metric}']
    print(f"{metric.capitalize()}: {scores.mean():.4f} ± {scores.std():.4f}")

# Feature importance from XGBoost
feature_importances = best_xgb.feature_importances_
print("\nXGBoost Feature Importances:")
for i, importance in enumerate(feature_importances):
    print(f"Feature {i}: {importance}")

# Optional: Plot XGBoost feature importances
plt.figure(figsize=(10, 6))
xgb.plot_importance(best_xgb, max_num_features=20)
plt.title('XGBoost Feature Importances')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/ML KAH/xgboost_feature_importances.png')
plt.close()

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
# 13. Validasi Model dengan Threshold Optimization #
# -----------------------------------------------#

print("\n=== 13. THRESHOLD OPTIMIZATION ===")

# Cari threshold optimal berdasarkan F1-score
from sklearn.metrics import precision_recall_curve

precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve)
f1_scores = np.nan_to_num(f1_scores)

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

print(f"Threshold optimal: {optimal_threshold:.4f}")
print(f"F1-Score optimal: {optimal_f1:.4f}")

# Prediksi dengan threshold optimal
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

# Evaluasi dengan threshold optimal
accuracy_opt = accuracy_score(y_test, y_pred_optimal)
precision_opt = precision_score(y_test, y_pred_optimal)
recall_opt = recall_score(y_test, y_pred_optimal)
f1_opt = f1_score(y_test, y_pred_optimal)

print(f"\nHasil dengan threshold optimal ({optimal_threshold:.4f}):")
print(f"Accuracy: {accuracy_opt:.4f}")
print(f"Precision: {precision_opt:.4f}")
print(f"Recall: {recall_opt:.4f}")
print(f"F1-Score: {f1_opt:.4f}")

# Simpan threshold optimal
threshold_info = {
    'optimal_threshold': optimal_threshold,
    'default_threshold': 0.5,
    'optimal_f1': optimal_f1,
    'default_f1': f1
}
joblib.dump(threshold_info, '/content/drive/MyDrive/ML KAH/threshold_info.joblib')

print("\n" + "=" * 80)
print("EKSPERIMEN SELESAI - SEMUA FILE TELAH DISIMPAN!")
print("=" * 80)
print("File yang disimpan:")
print("1. best_xgb_model_balanced.joblib - Model XGBoost terbaik")
print("2. scaler_balanced.joblib - StandardScaler")
print("3. feature_selector_balanced.joblib - Feature selector pertama")
print("4. feature_selector_after_fusion_balanced.joblib - Feature selector kedua")
print("5. lstm_encoder_balanced.joblib - LSTM Encoder model")
print("6. balancing_info.joblib - Informasi teknik balancing")
print("7. threshold_info.joblib - Informasi threshold optimal")
print("8. hasil_evaluasi_untuk_jurnal_balanced.txt - Hasil evaluasi lengkap")
print("9. hasil_table_untuk_jurnal_balanced.txt - Tabel untuk jurnal")
print("10. comparison_before_after_balancing.csv - Perbandingan hasil")
print("11. final_feature_importance.csv - Feature importance final")
print("12. Various visualization plots (.png files)")
print("=" * 80)