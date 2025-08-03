# ==============================================================================
# PEMBUATAN MODEL (OTAK) UNTUK PREDIKSI RISIKO KREDIT
# ==============================================================================
# Import library yang dibutuhkan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import joblib 
import os 
import matplotlib.pyplot as plt

print("Proses Pelatihan Model Dimulai...")

# --- LANGKAH 0: PERSIAPAN FOLDER ---
# Membuat folder 'model' jika belum ada, untuk menyimpan hasil
model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Folder '{model_dir}' berhasil dibuat.")

# --- LANGKAH 1: MEMUAT DATASET ---
# Path disesuaikan dengan struktur folder Anda
dataset_path = '/Users/mac/decision-tree/dataset/vehicle_credit_risk_dataset.xlsx'
try:
    # Menggunakan pd.read_excel karena file Anda adalah .xlsx
    df = pd.read_excel(dataset_path)
    print("\n1. Dataset berhasil dimuat dari:", dataset_path)
    print(f"   - Jumlah baris: {df.shape[0]}")
    print(f"   - Jumlah kolom: {df.shape[1]}")
except FileNotFoundError:
    print(f"Error: File '{dataset_path}' tidak ditemukan.")
    print("Pastikan file dataset berada di dalam folder 'dataset' sesuai struktur Anda.")
    exit()

# --- LANGKAH 2: PERSIAPAN DATA (PREPROCESSING) ---
print("\n2. Mempersiapkan data (Preprocessing)...")

df = df.drop('customer_id', axis=1)
X = df.drop('risk', axis=1)
y = df['risk']

categorical_cols = X.select_dtypes(include=['object']).columns
print(f"   - Kolom kategorikal yang akan diubah: {list(categorical_cols)}")

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
X_encoded.columns = encoder.get_feature_names_out(categorical_cols)

X_numeric = X.select_dtypes(include=['int64', 'float64'])
X_processed = pd.concat([X_numeric.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)


model_columns = X_processed.columns
print("   - Data berhasil di-preprocess. Jumlah fitur setelah encoding:", len(model_columns))


# --- LANGKAH 3: PEMBAGIAN DATA ---
# Membagi data menjadi 80% untuk latihan (training) dan 20% untuk pengujian (testing)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
print("\n3. Data berhasil dibagi menjadi data latih dan data uji.")
print(f"   - Jumlah data latih: {len(X_train)}")
print(f"   - Jumlah data uji: {len(X_test)}")


# --- LANGKAH 4: PELATIHAN MODEL DECISION TREE ---
print("\n4. Melatih model Decision Tree...")
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)
print("   - Model berhasil dilatih!")


# --- LANGKAH 5: EVALUASI MODEL ---
print("\n5. Mengevaluasi performa model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n   - Akurasi Model: {accuracy * 100:.2f}%")
print("\n   - Laporan Klasifikasi Rinci:")
print(classification_report(y_test, y_pred))


# --- LANGKAH 6: MENYIMPAN MODEL (OTAK) ---
print("\n6. Menyimpan 'otak' (model) yang sudah dilatih...")
joblib.dump(model, os.path.join(model_dir, 'model_risiko_kredit.joblib'))
joblib.dump(model_columns, os.path.join(model_dir, 'kolom_model.joblib'))
print(f"   - Model berhasil disimpan di folder '{model_dir}'.")

# --- LANGKAH 7: MEMBUAT DAN MENYIMPAN VISUALISASI POHON TREE ---
print("\n7. Membuat visualisasi pohon keputusan...")
try:
    # Membuat folder 'assets' di dalam 'frontend' jika belum ada
    assets_dir = os.path.join('frontend', 'assets')
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    plt.figure(figsize=(40, 20)) # Membuat gambar yang besar agar muat
    plot_tree(model, 
              feature_names=model_columns, 
              class_names=model.classes_, 
              filled=True, 
              rounded=True, 
              fontsize=8,
              max_depth=4) # Batasi kedalaman pohon yang digambar agar tidak terlalu ramai
    
    output_path = os.path.join(assets_dir, 'pohon_keputusan.png')
    plt.savefig(output_path)
    plt.close()
    print(f"   - Visualisasi pohon berhasil disimpan di: {output_path}")

except Exception as e:
    print(f"   - Gagal membuat visualisasi: {e}")


print("\nProses Selesai!")