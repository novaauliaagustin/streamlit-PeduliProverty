import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ------------------------------------
# Dummy model (to replace with actual model)
# ------------------------------------
# Generate sample dataset for demonstration purposes
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model (Replacing GradientBoosting with RandomForest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ------------------------------------
# Streamlit UI Implementation
# ------------------------------------
st.image("logo.png", width=150)
st.title("PeduliPoverty: Platform Berbasis Data untuk Optimalisasi Program Bantuan Pemerintah")
st.write("""
PeduliPoverty adalah platform berbasis data yang dirancang untuk mendeteksi dan menganalisis tingkat kemiskinan guna mendukung pemerintah dalam menyalurkan bantuan secara lebih tepat sasaran.
Dengan memanfaatkan teknologi dan analisis data, platform ini bertujuan untuk memastikan bahwa program bantuan sosial pemerintah tepat sasaran, sehingga dapat mengurangi kesenjangan sosial dan memberikan bantuan kepada mereka yang benar-benar membutuhkan.
""")
st.write("Mohon isi data berikut untuk mendapatkan hasil prediksi:")

# Input user melalui Streamlit
def user_input_features():
    name = st.text_input("Nama")
    age = st.number_input("Umur", min_value=0, max_value=120, step=1)
    gender = st.radio("Jenis Kelamin", ("Laki-laki", "Perempuan"))
    married = st.radio("Status Pernikahan", ("Belum Menikah", "Sudah Menikah"))
    education_level = st.selectbox(
        "Tingkat Pendidikan",
        ("Tidak Sekolah", "SD", "SMP", "SMA", "Perguruan Tinggi")
    )
    employed_last_year = st.radio("Pekerjaan dalam setahun terakhir", ("Tidak", "Ya"))
    income_own_business_last_year = st.radio("Pendapatan dari Bisnis Sendiri", ("Tidak", "Ya"))
    formal_savings = st.radio("Memiliki Tabungan Formal", ("Tidak", "Ya"))
    num_times_borrowed_last_year = st.number_input("Jumlah Pinjaman dalam Setahun Terakhir", min_value=0, step=1)
    reg_bank_acct = st.radio("Memiliki Akun Bank", ("Tidak", "Ya"))
    can_make_transaction = st.radio("Bisa Menggunakan Layanan Keuangan Digital", ("Tidak", "Ya"))

    data = {
        'age': age,
        'female': 1 if gender == "Perempuan" else 0,
        'married': 1 if married == "Sudah Menikah" else 0,
        'education_level': ["Tidak Sekolah", "SD", "SMP", "SMA", "Perguruan Tinggi"].index(education_level),
        'employed_last_year': 1 if employed_last_year == "Ya" else 0,
        'income_own_business_last_year': 1 if income_own_business_last_year == "Ya" else 0,
        'formal_savings': 1 if formal_savings == "Ya" else 0,
        'num_times_borrowed_last_year': num_times_borrowed_last_year,
        'reg_bank_acct': 1 if reg_bank_acct == "Ya" else 0,
        'can_make_transaction': 1 if can_make_transaction == "Ya" else 0
    }
    return name, pd.DataFrame([data])

# Mendapatkan input pengguna
name, input_df = user_input_features()

# Tombol untuk melakukan prediksi
if st.button("Lihat Hasil Prediksi"):
    try:
        # Pastikan fitur input sesuai dengan model
        features_for_prediction = input_df.reindex(columns=[
            'age', 'female', 'married', 'education_level', 'employed_last_year',
            'income_own_business_last_year', 'formal_savings',
            'num_times_borrowed_last_year', 'reg_bank_acct', 'can_make_transaction'
        ], fill_value=0)

        # Melakukan prediksi
        prediction = model.predict(features_for_prediction)

        # Menampilkan hasil prediksi
        if prediction[0] == 0:
            st.success(f"""
        Hasil Prediksi:
        Atas nama {name} dikategorikan TIDAK MISKIN.
        Mohon catat hasil ini dalam sistem dan pastikan data telah tersimpan dengan baik untuk keperluan administrasi.
        Tidak diperlukan langkah lebih lanjut terkait bantuan sosial untuk individu ini.
        """)        
        else:
            st.markdown(f"""
        <div style="background-color:#d4edda;padding:10px;border-radius:5px">
        <h4 style="color:#155724;">Hasil Prediksi:</h4>
        <p style="color:#155724;">Atas nama <b>{name}</b> dikategorikan <b>TIDAK MISKIN</b>.</p>
        <p style="color:#155724;">Mohon catat hasil ini dalam sistem dan pastikan data telah tersimpan dengan baik untuk keperluan administrasi.
        Tidak diperlukan langkah lebih lanjut terkait bantuan sosial untuk individu ini.</p>
        </div>
        """, unsafe_allow_html=True)     
    except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")