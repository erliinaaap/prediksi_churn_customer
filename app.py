import streamlit as st
import pandas as pd
import joblib

# ── Load model & artifacts ──────────────────────────────────────────
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ── Halaman utama ───────────────────────────────────────────────────
st.set_page_config(
    page_title='Customer Churn Predictor',
    page_icon='📊',
    layout='wide'
)

st.title('📊 Customer Churn Prediction')
st.markdown("""
Aplikasi ini memprediksi apakah seorang pelanggan akan **churn**
berdasarkan data aktivitas dan demografis mereka.
""")

st.divider()
st.subheader('🔧 Input Data Pelanggan')

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Usia (age)', min_value=18, max_value=90, value=35)
    total_visits = st.number_input('Total Kunjungan', min_value=0, max_value=1000, value=10)
    avg_session_time = st.number_input('Avg Session Time (menit)', min_value=0.0, max_value=60.0, value=10.0)
    total_spent = st.number_input('Total Pengeluaran (USD)', min_value=0.0, max_value=10000.0, value=500.0)
    satisfaction_score = st.slider('Satisfaction Score', 1.0, 5.0, 3.0, 0.1)

with col2:
    nps_score = st.number_input('NPS Score', min_value=-100, max_value=100, value=0)
    lifetime_value = st.number_input('Lifetime Value', min_value=0.0, max_value=10000.0, value=1000.0)
    last_3_month_purchase_freq = st.number_input('Frekuensi Beli 3 Bulan', min_value=0, max_value=50, value=3)
    tenure_days = st.number_input('Lama Berlangganan (hari)', min_value=0, max_value=3000, value=365)
    days_since_purchase = st.number_input('Hari Sejak Pembelian Terakhir', min_value=0, max_value=1000, value=30)

    # FITUR YANG HILANG
    days_since_signup = st.number_input(
        'Hari Sejak Registrasi',
        min_value=0,
        max_value=5000,
        value=365
    )

    support_tickets = st.number_input(
        'Jumlah Tiket Support',
        min_value=0,
        max_value=100,
        value=0
    )

with col3:
    email_open_rate = st.slider('Email Open Rate', 0.0, 1.0, 0.3)
    email_click_rate = st.slider('Email Click Rate', 0.0, 1.0, 0.1)
    pages_per_session = st.number_input('Halaman per Sesi', min_value=0.0, max_value=30.0, value=5.0)
    avg_order_value = st.number_input('Rata-rata Nilai Transaksi', min_value=0.0, max_value=2000.0, value=100.0)
    marketing_spend_per_user = st.number_input('Marketing Spend per User', min_value=0.0, max_value=500.0, value=20.0)

# ── Tombol Prediksi ─────────────────────────────────────────────────
st.divider()

if st.button('🔍 Prediksi Churn', use_container_width=True, type="primary"):

    input_data = {
        'age': age,
        'total_visits': total_visits,
        'avg_session_time': avg_session_time,
        'total_spent': total_spent,
        'satisfaction_score': satisfaction_score,
        'nps_score': nps_score,
        'lifetime_value': lifetime_value,
        'last_3_month_purchase_freq': last_3_month_purchase_freq,
        'tenure_days': tenure_days,
        'days_since_purchase': days_since_purchase,
        'days_since_signup': days_since_signup,
        'support_tickets': support_tickets,
        'email_open_rate': email_open_rate,
        'email_click_rate': email_click_rate,
        'pages_per_session': pages_per_session,
        'avg_order_value': avg_order_value,
        'marketing_spend_per_user': marketing_spend_per_user
    }

    input_df = pd.DataFrame([input_data])

    # Pastikan semua fitur yang dibutuhkan model tersedia
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Urutan kolom harus sama seperti saat training
    input_selected = input_df[features]

    input_scaled = scaler.transform(input_selected)

    pred = model.predict(input_scaled)[0]

    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_scaled)[0]

    st.subheader("📊 Hasil Prediksi")

    if pred == 1:
        st.error("⚠️ Pelanggan ini BERPOTENSI CHURN")
        if prob is not None:
            st.metric("Probabilitas Churn", f"{prob[1]*100:.2f}%")
    else:
        st.success("✅ Pelanggan ini TIDAK BERPOTENSI CHURN")
        if prob is not None:
            st.metric("Probabilitas Bertahan", f"{prob[0]*100:.2f}%")

    with st.expander("📋 Detail Input"):
        st.dataframe(input_df)