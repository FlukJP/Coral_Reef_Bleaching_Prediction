import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

# ─── Load Models ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    sc_roc   = joblib.load("models/scaler_ROC.pkl")
    nn_roc   = keras.models.load_model("models/neural_network_ROC.keras")

    sc_gbe       = joblib.load("models/scaler_GBE.pkl")
    poly_gbe     = joblib.load("models/poly_GBE.pkl")
    thresh_gbe   = joblib.load("models/threshold_GBE.pkl")
    gbe_s1       = keras.models.load_model("models/stage1_classifier_GBE.keras")
    gbe_s2       = keras.models.load_model("models/stage2_regressor_GBE.keras")

    return sc_roc, nn_roc, sc_gbe, poly_gbe, thresh_gbe, gbe_s1, gbe_s2


def show():
    try:
        sc_roc, nn_roc, sc_gbe, poly_gbe, thresh_gbe, gbe_s1, gbe_s2 = load_all_models()
    except Exception as e:
        st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
        st.stop()

    # ─── Header ──────────────────────────────────────────────────────────────────
    st.title("🪸 ระบบพยากรณ์ความเสี่ยงปะการังฟอกขาว")
    st.caption("ใช้ Neural Network วิเคราะห์สภาพแวดล้อมทางทะเลจาก 2 ชุดข้อมูล")
    st.divider()

    # ─── Tabs ─────────────────────────────────────────────────────────────────────
    tab_roc, tab_gbe = st.tabs([
        "🌊 ROC — จำแนกระดับความรุนแรง",
        "🌐 GBE — ทำนายเปอร์เซ็นต์การฟอกขาว",
    ])

    # TAB 1 — ROC  (Binary Classification: Low / Severe)
    with tab_roc:
        st.subheader("ชุดข้อมูล: Realistic Ocean Climate (ROC)")
        st.write("จำแนกความรุนแรงของการฟอกขาว → **Low (ต่ำ)** หรือ **Severe (รุนแรง)**")
        st.divider()

        st.markdown("#### กรอกข้อมูลสภาพแวดล้อม")
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Latitude (ละติจูด)",  value=0.0,  step=0.01, key="lat")
            sst = st.number_input("SST — อุณหภูมิน้ำทะเล (°C)", value=28.0, step=0.1, key="sst",
                                  help="Sea Surface Temperature ปกติอยู่ที่ 26–30 °C")
        with c2:
            lon = st.number_input("Longitude (ลองจิจูด)", value=0.0,  step=0.01, key="lon")
            ph  = st.number_input("pH Level — ระดับความกรดด่าง", value=8.1,  step=0.01, key="ph",
                                  help="ค่า pH ปกติของทะเลอยู่ที่ 7.9–8.3")

        if st.button("🔍 ทำนายความเสี่ยง", type="primary", key="btn_roc"):
            df_in = pd.DataFrame({
                "Latitude":      [lat],
                "Longitude":     [lon],
                "SST (°C)":      [sst],
                "pH Level":      [ph],
                "Abs_Latitude":  [abs(lat)],
                "SST_pH_Stress": [sst / ph if ph != 0 else 0.0],
            })
            X_sc = sc_roc.transform(df_in)

            prob_raw = nn_roc.predict(X_sc, verbose=0)[0][0]
            pred     = int(prob_raw > 0.53)
            prob     = float(prob_raw)

            st.divider()
            st.markdown("### ผลการทำนาย")
            col_res, col_prob = st.columns([2, 1])
            with col_res:
                if pred == 0:
                    st.success("🟢 **ระดับความเสี่ยง: ต่ำ (Low)**\n\nปะการังอยู่ในสภาวะค่อนข้างปลอดภัย")
                else:
                    st.error("🔴 **ระดับความเสี่ยง: รุนแรง (Severe)**\n\nมีความเสี่ยงสูงที่ปะการังจะเกิดการฟอกขาว!")
            with col_prob:
                st.metric("ความน่าจะเป็น (Severe)", f"{prob:.1%}")

    # TAB 2 — GBE  (Regression: Percent Bleaching 0–100)
    with tab_gbe:
        st.subheader("ชุดข้อมูล: Global Bleaching Events (GBE)")
        st.write("ทำนาย **เปอร์เซ็นต์การฟอกขาว** (0 – 100%)")
        st.divider()

        st.markdown("#### กรอกข้อมูลสภาพแวดล้อม")
        g1, g2 = st.columns(2)
        with g1:
            depth     = st.number_input("Depth (ความลึก, เมตร)",          value=10.0,  step=0.5,   key="depth")
            turbidity = st.number_input("Turbidity (ความขุ่น)",            value=0.03,  step=0.001, format="%.4f", key="turb")
            cyc_freq  = st.number_input("Cyclone Frequency (ความถี่พายุ)", value=50.0,  step=0.1,   key="cyc")
            ssta_dhw  = st.number_input("SSTA DHW (ค่าความร้อนสะสม)",      value=0.0,   step=0.01,  key="dhw",
                                        help="Degree Heating Weeks — ค่าสูงบ่งชี้ความเครียดความร้อนสะสม")
            ssta      = st.number_input("SSTA (Sea Surface Temp Anomaly)",  value=0.0,   step=0.01,  key="ssta")
        with g2:
            dist_shore = st.number_input("Distance to Shore (เมตร)",       value=1000.0, step=10.0, key="dist")
            wind       = st.number_input("Windspeed (ความเร็วลม, m/s)",   value=5.0,    step=0.1,  key="wind")
            temp_max   = st.number_input("Temperature Maximum (K)",         value=304.0,  step=0.1,  key="tmax",
                                         help="อุณหภูมิสูงสุดในหน่วย Kelvin (300 K ≈ 27 °C)")
            ssta_max   = st.number_input("SSTA Maximum",                    value=2.0,    step=0.01, key="ssta_max")
            tsa_dhw    = st.number_input("TSA DHW",                         value=0.0,    step=0.01, key="tsa_dhw")

        if st.button("🔍 ทำนายเปอร์เซ็นต์การฟอกขาว", type="primary", key="btn_gbe"):
            df_in = pd.DataFrame({
                "Depth_m":             [depth],
                "Distance_to_Shore":   [dist_shore],
                "Turbidity":           [turbidity],
                "Windspeed":           [wind],
                "Cyclone_Frequency":   [cyc_freq],
                "Temperature_Maximum": [temp_max],
                "SSTA":                [ssta],
                "SSTA_Maximum":        [ssta_max],
                "SSTA_DHW":            [ssta_dhw],
                "TSA_DHW":             [tsa_dhw],
            })
            X_sc   = sc_gbe.transform(df_in)
            X_poly = poly_gbe.transform(X_sc)

            # Stage 1 — จำแนกว่ามีการฟอกขาวหรือไม่
            stage1_prob   = float(gbe_s1.predict(X_poly, verbose=0)[0][0])
            has_bleaching = stage1_prob > thresh_gbe

            # Stage 2 — ทำนายเปอร์เซ็นต์เฉพาะเมื่อ Stage 1 บอกว่ามีการฟอกขาว
            if has_bleaching:
                pred_sqrt = gbe_s2.predict(X_poly, verbose=0)[0][0]
                pct = float(np.clip(np.square(pred_sqrt), 0, 100))
            else:
                pct = 0.0

            st.divider()
            st.markdown("### ผลการทำนาย")

            col_metric, col_bar = st.columns([1, 2])
            with col_metric:
                st.metric("เปอร์เซ็นต์การฟอกขาว", f"{pct:.1f}%")
                st.caption(f"Stage 1 (มีการฟอกขาว): {stage1_prob:.1%}  (threshold={thresh_gbe:.2f})")
            with col_bar:
                if pct < 25:
                    level, color = "Low", "🟢"
                elif pct < 50:
                    level, color = "Moderate", "🟡"
                elif pct < 75:
                    level, color = "High", "🟠"
                else:
                    level, color = "Critical", "🔴"
                st.markdown(f"**{color} ระดับ: {level}**")
                st.progress(pct / 100)
