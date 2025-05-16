import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Dropout

# ----------------#
# 1. IMPORT DATA  #
# ----------------#
data = pd.read_csv('dataset/cdc_diabetes_health_indicators.csv')
print("Data shape:", data.shape)
print("\nSample data:")
print(data.head())
print("\nData info:")
print(data.info())
print("\nStatistik deskriptif:")
print(data.describe())

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
rf = RandomForestClassifier(n_estimators=100, random_state=42)
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
plt.tight_layout()
plt.savefig('feature_importances.png')
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

# -------------------------------------------------#
# 5. Optional: LSTM Autoencoder Feature Extraction #
# -------------------------------------------------#

print("\n=== 5. LSTM Autoencoder untuk Feature Extraction ===")
#  Reshape data untuk LSTM [samples, timesteps, features]
timesteps = 1
X_train_lstm = X_train.reshape(X_train.shape[0], timesteps, X_train.shape[1])
X_test_lstm = X_test.reshape(X_test.shape[0], timesteps, X_test.shape[1])

# Buat model LSTM Autoencoder
input_dim = X_train.shape[1]
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
    X_train_lstm, X_train_lstm,
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
plt.savefig('autoencoder_loss.png')
plt.close()

# Extract features using encoder
X_train_encoded = encoder_model.predict(X_train_lstm)
X_test_encoded = encoder_model.predict(X_test_lstm)

print(f"Encoded features shape: {X_train_encoded.shape}")


# ----------------------------------------------------------#
# 6. Feature Fusion (PCA + Random Forest selected features) #
# ----------------------------------------------------------#

print("\n=== 6. Feature Fusion (PCA + Random Forest selected features) ===")
# Implementasi feature fusion dengan Original features + Encoded features
X_train_original = X_train
X_test_original = X_test

# Gabungkan fitur asli dengan fitur hasil encoding
X_train_fused = np.hstack((X_train_original, X_train_encoded))
X_test_fused = np.hstack((X_test_original, X_test_encoded))

print(f"Feature fusion shape: {X_train_fused.shape}")

# Custom Feature Fusion untuk fitur tertentu
X_train_df = pd.DataFrame(X_train, columns=selected_features)
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
# 7. Feature Selection Ulang setelah Feature Fusion #
# --------------------------------------------------#

print("\n=== 7. Feature Selection Ulang setelah Feature Fusion ===")
rf_after_fusion = RandomForestClassifier(n_estimators=100, random_state=42)
rf_after_fusion.fit(X_train_fused, y_train)

# Pilih fitur lagi setelah fusion
selector_after_fusion = SelectFromModel(rf_after_fusion, threshold='mean', prefit=True)
X_train_final = selector_after_fusion.transform(X_train_fused)
X_test_final = selector_after_fusion.transform(X_test_fused)

print(f"Jumlah fitur final: {X_train_final.shape[1]} dari {X_train_fused.shape[1]}")

# --------------------------------------------------#
# 8. Train Final Model (XGBoost + Cross Validation) #
# --------------------------------------------------#

print("\n=== 8. Train Final Model (XGBoost + Cross Validation) ===")

# Import XGBoost
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score

# Parameter tuning untuk XGBoost
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1]
}

# Grid search with cross-validation
xgb_grid = GridSearchCV(
    xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='logloss'),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

xgb_grid.fit(X_train_final, y_train)

# Best model parameters
print("Best parameters:", xgb_grid.best_params_)
print("Best cross-validation score: {:.4f}".format(xgb_grid.best_score_))

# Get the best model
best_xgb = xgb_grid.best_estimator_

# Cross-validation with the best model
cv_scores = cross_val_score(best_xgb, X_train_final, y_train, cv=5, scoring='accuracy')
print("Cross-validation accuracy: {:.4f} ± {:.4f}".format(cv_scores.mean(), cv_scores.std()))

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
plt.savefig('xgboost_feature_importances.png')
plt.close()

 # ------------------#
 # 9. Final Testing  #
 # ------------------#

# 9. Final Testing
print("\n=== 9. Final Testing ===")
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

# Tampilkan hasil metrik untuk publikasi jurnal
print("\n" + "="*50)
print("HASIL EVALUASI MODEL UNTUK PUBLIKASI JURNAL")
print("="*50)
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1-Score      : {f1:.4f}")
print(f"ROC AUC       : {roc_auc_value:.4f}")
print(f"MCC           : {mcc:.4f}")
print(f"Cohen's Kappa : {kappa:.4f}")
print(f"Avg Precision : {average_precision:.4f}")
print("="*50)

# Simpan hasil untuk jurnal
with open('hasil_evaluasi_untuk_jurnal.txt', 'w') as f:
    f.write("HASIL EVALUASI MODEL UNTUK PUBLIKASI JURNAL\n")
    f.write("="*50 + "\n")
    f.write(f"Accuracy      : {accuracy:.4f}\n")
    f.write(f"Precision     : {precision:.4f}\n")
    f.write(f"Recall        : {recall:.4f}\n")
    f.write(f"F1-Score      : {f1:.4f}\n")
    f.write(f"ROC AUC       : {roc_auc_value:.4f}\n")
    f.write(f"MCC           : {mcc:.4f}\n")
    f.write(f"Cohen's Kappa : {kappa:.4f}\n")
    f.write(f"Avg Precision : {average_precision:.4f}\n")
    f.write("="*50 + "\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix dengan persentase
cm = confusion_matrix(y_test, y_pred)
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
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.savefig('model_evaluation_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

# Tambahkan visualisasi XGBoost tree (optional)
plt.figure(figsize=(15, 10))
xgb.plot_tree(best_xgb, num_trees=0)
plt.title('First Tree in XGBoost Model')
plt.tight_layout()
plt.savefig('xgboost_tree.png')
plt.close()

# Simpan model terbaik
import joblib
import pandas as pd
from datetime import datetime

# Simpan model
joblib.dump(best_xgb, 'best_xgb_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(selector, 'feature_selector.joblib')
joblib.dump(selector_after_fusion, 'feature_selector_after_fusion.joblib')
joblib.dump(encoder_model, 'lstm_encoder.joblib')

# Simpan hasil eksperimen untuk jurnal
experiment_results = {
    'Model': 'XGBoost + LSTM Autoencoder + Feature Fusion',
    'Tanggal': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'Dataset_Size': X.shape[0],
    'Original_Features': X.shape[1],
    'Selected_Features': len(selected_features),
    'Final_Features': X_train_final.shape[1],
    'Best_Parameters': xgb_grid.best_params_,
    'CV_Accuracy_Mean': cv_scores.mean(),
    'CV_Accuracy_Std': cv_scores.std(),
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
pd.DataFrame([experiment_results]).to_csv('hasil_eksperimen_untuk_jurnal.csv', index=False)

# Simpan hasil dengan format tabel untuk jurnal
with open('hasil_table_untuk_jurnal.txt', 'w') as f:
    f.write("Table X: Performance metrics of the proposed hybrid model for diabetes prediction\n\n")
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
    f.write("*Note: The proposed model combines XGBoost with LSTM Autoencoder feature extraction and custom feature fusion techniques.")

print("\nProses selesai! Model terbaik tersimpan sebagai 'best_xgb_model.joblib'")

# Buat fungsi untuk prediksi data baru
def predict_diabetes(new_data):
    """
    Memprediksi diabetes pada data baru

    Parameters:
    -----------
    new_data : pandas DataFrame
        Data baru yang akan diprediksi, harus memiliki kolom yang sama dengan dataset asli

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
    if 'Diabetes_binary' in new_data.columns:
        new_data = new_data.drop('Diabetes_binary', axis=1)

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

    return prediction, probability

print("\nContoh penggunaan fungsi prediksi:")
print("prediction, probability = predict_diabetes(new_patient_data)")