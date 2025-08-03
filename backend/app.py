# ==============================================================================
# BACKEND (FLASK) UNTUK APLIKASI PREDIKSI RISIKO KREDIT
# Versi ini disempurnakan dengan fitur Prediksi Massal dan Penjelasan Prediksi.
# ==============================================================================
# Import library yang dibutuhkan
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os
import traceback
import numpy as np

# --- LANGKAH 0: KONFIGURASI PATH ---
base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_folder = os.path.join(base_dir, '..', 'frontend')
app = Flask(__name__, static_folder=frontend_folder)
CORS(app)

# --- LANGKAH 1: MEMUAT "OTAK" (MODEL) YANG SUDAH DILATIH ---
try:
    model_path = os.path.join(base_dir, '..', 'model', 'model_risiko_kredit.joblib')
    columns_path = os.path.join(base_dir, '..', 'model', 'kolom_model.joblib')
    
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
    
    print(">>> Model dan daftar kolom berhasil dimuat.")
except FileNotFoundError:
    print(">>> FATAL ERROR: File model tidak ditemukan.")
    print(f">>> Mencari di path: {model_path}")
    print(">>> Pastikan Anda sudah menjalankan 'latih.py' dan file model ada di folder 'model'.")
    model = None
    model_columns = None


# --- LANGKAH 2: FUNGSI PREDIKSI DAN PENJELASAN ---
def preprocess_data(df_input):
    """Fungsi terpusat untuk memproses data mentah menjadi format yang siap diprediksi."""
    categorical_cols = [col for col in df_input.columns if df_input[col].dtype == 'object']
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols, dtype=int)
    df_aligned = df_encoded.reindex(columns=model_columns, fill_value=0)
    return df_aligned

def get_top_features(model, columns):
    """Mendapatkan fitur terpenting dari model secara global."""
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': columns, 'importance': importances})
    # Mengurutkan berdasarkan kepentingan dan mengambil 3 teratas
    top_features = feature_importance_df.sort_values(by='importance', ascending=False).head(3)
    return top_features['feature'].tolist()

# --- LANGKAH 3: ENDPOINT API ---

# Endpoint untuk Prediksi Tunggal
@app.route('/predict', methods=['POST'])
def handle_prediction():
    if not request.is_json:
        return jsonify({"error": "Request harus dalam format JSON"}), 400
        
    input_data = request.get_json()
    
    try:
        df_input = pd.DataFrame([input_data])
        processed_df = preprocess_data(df_input)
        
        # Melakukan prediksi
        prediction = model.predict(processed_df)
        risiko = prediction[0]
        
        # Mendapatkan fitur penentu utama
        top_features = get_top_features(model, model_columns)
        
        alasan = f"Berdasarkan analisis data, nasabah masuk dalam kategori risiko '{risiko}'."

        return jsonify({
            "risiko": risiko,
            "alasan": alasan,
            "faktor_penentu": top_features # Mengirimkan faktor penentu ke frontend
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Terjadi kesalahan saat melakukan prediksi tunggal."}), 500

# Endpoint Baru untuk Prediksi Massal
@app.route('/batch-predict', methods=['POST'])
def handle_batch_prediction():
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nama file tidak boleh kosong."}), 400

    try:
        # Membaca file yang diunggah (bisa CSV atau Excel)
        if file.filename.endswith('.csv'):
            df_batch = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df_batch = pd.read_excel(file)
        else:
            return jsonify({"error": "Format file tidak didukung. Harap unggah .csv atau .xlsx"}), 400

        # Simpan data asli untuk ditampilkan kembali
        original_data = df_batch.to_dict(orient='records')

        # Hapus kolom ID jika ada
        if 'customer_id' in df_batch.columns:
            df_batch = df_batch.drop('customer_id', axis=1)

        # Preprocess data
        processed_df_batch = preprocess_data(df_batch)
        
        # Lakukan prediksi untuk semua baris
        predictions = model.predict(processed_df_batch)
        
        # Gabungkan hasil prediksi dengan data asli
        results = []
        for i, record in enumerate(original_data):
            record['predicted_risk'] = predictions[i]
            results.append(record)

        return jsonify({"results": results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Terjadi kesalahan saat memproses file: {e}"}), 500


# --- LANGKAH 4: MENYAJIKAN FRONTEND ---
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(app.static_folder, filename)


# Menjalankan server
if __name__ == '__main__':
    app.run(debug=True, port=5005)