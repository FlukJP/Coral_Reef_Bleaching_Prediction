import streamlit as st
import numpy as np
import pandas as pd
import joblib


@st.cache_resource
def load_all_models():
    sc_roc   = joblib.load("models/scaler_stacking_ROC.pkl")
    clf_roc  = joblib.load("models/stacking_classifier_ROC.pkl")

    sc_gbe   = joblib.load("models/scaler_stacking_GBE.pkl")
    reg_gbe  = joblib.load("models/stacking_regressor_GBE.pkl")

    return sc_roc, clf_roc, sc_gbe, reg_gbe


# ── Label maps ───────────────────────────────────────────────────────────────
SEVERITY_LABELS = {
    0: ("🟢 ต่ำ (Low)",      "success", "ปะการังอยู่ในสภาวะปลอดภัย ความเสี่ยงต่ำมาก"),
    1: ("🟡 ปานกลาง (Medium)", "warning", "มีความเสี่ยงระดับปานกลาง ควรติดตามสถานการณ์"),
    2: ("🔴 รุนแรง (High)",   "error",   "ความเสี่ยงสูงมาก! ปะการังอาจเกิดการฟอกขาวรุนแรง"),
}


def _severity_badge(pred: int, proba: np.ndarray) -> None:
    label, level, desc = SEVERITY_LABELS[pred]
    getattr(st, level)(f"**{label}**\n\n{desc}")

    st.markdown("#### ความน่าจะเป็นของแต่ละระดับ")
    cols = st.columns(3)
    names = ["Low (0)", "Medium (1)", "High (2)"]
    for i, (col, name, p) in enumerate(zip(cols, names, proba)):
        col.metric(name, f"{p:.1%}", delta=None)
        col.progress(float(p))


def show():
    try:
        sc_roc, clf_roc, sc_gbe, reg_gbe = load_all_models()
    except Exception as e:
        st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
        st.info("กรุณารัน `notebook/train_stacking.ipynb` เพื่อสร้างโมเดลก่อน")
        st.stop()

    # ── Header ───────────────────────────────────────────────────────────────
    st.title("🪸 ระบบพยากรณ์ความเสี่ยงปะการังฟอกขาว")
    st.caption("ใช้ **Stacking Ensemble** (RF + SVM + KNN → Meta-Learner) วิเคราะห์จาก 2 ชุดข้อมูล")
    st.divider()

    tab_roc, tab_gbe = st.tabs([
        "🌊 ROC — จำแนกระดับความรุนแรง",
        "🌐 GBE — ทำนายเปอร์เซ็นต์การฟอกขาว",
    ])

    # ── TAB 1: ROC ────────────────────────────────────────────────────────────
    with tab_roc:
        st.subheader("ชุดข้อมูล: Realistic Ocean Climate (ROC)")
        st.write("จำแนกระดับความรุนแรงของการฟอกขาว → **Low / Medium / High**")
        st.divider()

        st.markdown("#### กรอกข้อมูลสภาพแวดล้อม")
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input(
                "Latitude (ละติจูด)", value=0.0, step=0.01,
                min_value=-90.0, max_value=90.0, key="roc_lat"
            )
            sst = st.number_input(
                "SST — อุณหภูมิน้ำทะเล (°C)", value=28.0, step=0.1,
                help="Sea Surface Temperature ปกติอยู่ที่ 26–30 °C", key="roc_sst"
            )
        with c2:
            lon = st.number_input(
                "Longitude (ลองจิจูด)", value=0.0, step=0.01,
                min_value=-180.0, max_value=180.0, key="roc_lon"
            )
            ph = st.number_input(
                "pH Level — ระดับความกรดด่าง", value=8.1, step=0.01,
                min_value=0.0, max_value=14.0,
                help="ค่า pH ปกติของทะเลอยู่ที่ 7.9–8.3", key="roc_ph"
            )

        if st.button("🔍 ทำนายระดับความรุนแรง", type="primary", key="btn_roc"):
            df_in = pd.DataFrame({
                "Latitude":  [lat],
                "Longitude": [lon],
                "SST (°C)":  [sst],
                "pH Level":  [ph],
            })
            X_sc   = sc_roc.transform(df_in)
            pred   = int(clf_roc.predict(X_sc)[0])
            proba  = clf_roc.predict_proba(X_sc)[0]

            st.divider()
            st.markdown("### ผลการทำนาย")

            colA, colB = st.columns([3, 2])
            with colA:
                _severity_badge(pred, proba)
            with colB:
                st.markdown("**ข้อมูลที่ใช้ทำนาย**")
                st.dataframe(df_in, use_container_width=True, hide_index=True)

    # ── TAB 2: GBE ────────────────────────────────────────────────────────────
    with tab_gbe:
        st.subheader("ชุดข้อมูล: Global Bleaching Events (GBE)")
        st.write("ทำนาย **เปอร์เซ็นต์การฟอกขาว** (0 – 100%)")
        st.divider()

        st.markdown("#### กรอกข้อมูลสภาพแวดล้อม")
        g1, g2 = st.columns(2)
        with g1:
            depth     = st.number_input("Depth (ความลึก, เมตร)",          value=10.0,   step=0.5,          key="gbe_depth")
            turbidity = st.number_input("Turbidity (ความขุ่น)",            value=0.03,   step=0.001, format="%.4f", key="gbe_turb")
            cyc_freq  = st.number_input("Cyclone Frequency (ความถี่พายุ)", value=50.0,   step=0.1,          key="gbe_cyc")
            ssta_dhw  = st.number_input(
                "SSTA DHW (ค่าความร้อนสะสม)", value=0.0, step=0.01, key="gbe_dhw",
                help="Degree Heating Weeks — ค่าสูงบ่งชี้ความเครียดความร้อนสะสม"
            )
        with g2:
            dist_shore = st.number_input("Distance to Shore (เมตร)",       value=1000.0, step=10.0,         key="gbe_dist")
            wind       = st.number_input("Windspeed (ความเร็วลม, m/s)",    value=5.0,    step=0.1,          key="gbe_wind")
            temp_max   = st.number_input(
                "Temperature Maximum (K)", value=304.0, step=0.1, key="gbe_tmax",
                help="อุณหภูมิสูงสุดในหน่วย Kelvin (300 K ≈ 27 °C)"
            )

        if st.button("🔍 ทำนายเปอร์เซ็นต์การฟอกขาว", type="primary", key="btn_gbe"):
            df_in = pd.DataFrame({
                "Depth_m":             [depth],
                "Distance_to_Shore":   [dist_shore],
                "Turbidity":           [turbidity],
                "Windspeed":           [wind],
                "Cyclone_Frequency":   [cyc_freq],
                "Temperature_Maximum": [temp_max],
                "SSTA_DHW":            [ssta_dhw],
            })
            X_sc      = sc_gbe.transform(df_in)
            pred_log  = reg_gbe.predict(X_sc)[0]
            pct       = float(np.clip(np.expm1(pred_log), 0, 100))

            st.divider()
            st.markdown("### ผลการทำนาย")

            colA, colB = st.columns([2, 3])
            with colA:
                st.metric("เปอร์เซ็นต์การฟอกขาว", f"{pct:.1f}%")

                if pct < 25:
                    level, color = "Low",      "🟢"
                elif pct < 50:
                    level, color = "Moderate", "🟡"
                elif pct < 75:
                    level, color = "High",     "🟠"
                else:
                    level, color = "Critical", "🔴"

                st.markdown(f"**{color} ระดับ: {level}**")
                st.progress(pct / 100)

            with colB:
                st.markdown("**ข้อมูลที่ใช้ทำนาย**")
                st.dataframe(df_in.T.rename(columns={0: "ค่า"}),
                            use_container_width=True)
