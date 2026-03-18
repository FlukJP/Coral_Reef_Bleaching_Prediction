import streamlit as st


def show():
    st.title("📖 รายละเอียดโมเดล")
    st.caption("อธิบายสถาปัตยกรรม ฟีเจอร์ และผลการเรียนรู้ของแต่ละโมเดล")
    st.divider()

    tab_ml, tab_nn = st.tabs(["🤖 Machine Learning (Stacking Ensemble)", "🧠 Neural Network (ANN)"])

    # ─── TAB 1: ML Stacking ───────────────────────────────────────────────────
    with tab_ml:
        st.header("Stacking Ensemble Classifier / Regressor")
        st.markdown(
            "โมเดล **Stacking** รวมความสามารถของหลายโมเดลเข้าด้วยกัน "
            "โดยให้ Base Models เรียนรู้ข้อมูลพร้อมกัน แล้วส่งผลทำนายให้ Meta-Learner "
            "\"ตัดสิน\" ว่าจะเชื่อโมเดลใดมากน้อยแค่ไหน"
        )

        st.subheader("สถาปัตยกรรม")
        st.code(
            """
ชั้นที่ 1 — Base Models (เรียนรู้ข้อมูลเดียวกัน)
  ├── 🌲 Random Forest      (Tree-based Ensemble)
  ├── 🔷 SVM / SVR          (Support Vector Machine / Regression)
  └── 📍 KNN                (K-Nearest Neighbors)
            │
            ▼  คำทำนายของ 3 โมเดล → กลายเป็น Features ชุดใหม่
ชั้นที่ 2 — Meta-Learner
  ├── Logistic Regression   (สำหรับ Classification — ROC)
  └── Linear Regression     (สำหรับ Regression — GBE)
            """,
            language="text",
        )

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("ROC — Classification")
            st.markdown("""
| รายการ | รายละเอียด |
|---|---|
| **ชุดข้อมูล** | Realistic Ocean Climate (ROC) |
| **Task** | Multiclass Classification |
| **Output** | 3 คลาส: Low / Medium / High |
| **Cross-Validation** | 5-Fold |
| **Scaler** | MinMaxScaler |
""")
            st.markdown("**Features ที่ใช้:**")
            st.markdown("""
- `Latitude` — ละติจูด
- `Longitude` — ลองจิจูด
- `SST (°C)` — อุณหภูมิผิวน้ำทะเล
- `pH Level` — ระดับความกรดด่าง
""")

        with col_right:
            st.subheader("GBE — Regression")
            st.markdown("""
| รายการ | รายละเอียด |
|---|---|
| **ชุดข้อมูล** | Global Bleaching Events (GBE) |
| **Task** | Regression |
| **Output** | เปอร์เซ็นต์การฟอกขาว (0–100%) |
| **Target Transform** | log1p ตอนเทรน, expm1 ตอนทำนาย |
| **Scaler** | MinMaxScaler |
""")
            st.markdown("**Features ที่ใช้:**")
            st.markdown("""
- `Depth_m` — ความลึก (เมตร)
- `Distance_to_Shore` — ระยะห่างจากชายฝั่ง
- `Turbidity` — ความขุ่นของน้ำ
- `Windspeed` — ความเร็วลม
- `Cyclone_Frequency` — ความถี่พายุหมุน
- `Temperature_Maximum` — อุณหภูมิสูงสุด (K)
- `SSTA_DHW` — ค่าความร้อนสะสม (Degree Heating Weeks)
""")

        st.divider()
        st.subheader("ผลเปรียบเทียบโมเดล")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown("**ROC — การเปรียบเทียบ Base Models vs Stacking**")
            try:
                st.image("models/stacking_roc_comparison.png", use_container_width=True)
            except Exception:
                st.info("ไม่พบไฟล์ภาพ `models/stacking_roc_comparison.png`")
        with img_col2:
            st.markdown("**GBE — การเปรียบเทียบ Base Models vs Stacking**")
            try:
                st.image("models/stacking_gbe_comparison.png", use_container_width=True)
            except Exception:
                st.info("ไม่พบไฟล์ภาพ `models/stacking_gbe_comparison.png`")

    # ─── TAB 2: Neural Network ────────────────────────────────────────────────
    with tab_nn:
        st.header("Artificial Neural Network (ANN)")
        st.markdown(
            "โมเดล **Neural Network** เลียนแบบการทำงานของเซลล์ประสาทในสมอง "
            "ประกอบด้วยหลาย Layer ที่เรียนรู้รูปแบบที่ซับซ้อนในข้อมูลโดยอัตโนมัติ"
        )

        st.subheader("สถาปัตยกรรม")

        col_nn_left, col_nn_right = st.columns(2)

        with col_nn_left:
            st.markdown("##### ROC — Binary Classification")
            st.code(
                """
Input Layer  (6 features)
      │
Dense(64) + Swish + BatchNorm + Dropout(0.3) + L2
      │
Dense(32) + Swish + BatchNorm + Dropout(0.2) + L2
      │
Dense(16) + Swish + L2
      │
Output(1)  + Sigmoid
      → threshold = 0.53
      → 0 = Low, 1 = Severe
                """,
                language="text",
            )
            st.markdown("""
| รายการ | รายละเอียด |
|---|---|
| **ชุดข้อมูล** | Realistic Ocean Climate (ROC) |
| **Task** | Binary Classification |
| **Output** | Low / Severe |
| **Loss** | Binary Crossentropy |
| **Optimizer** | Adam |
| **Scaler** | MinMaxScaler |
| **Best Threshold** | 0.53 (tuned via F1-macro) |
""")

        with col_nn_right:
            st.markdown("##### GBE — Two-Stage Regression")
            st.code(
                """
── Pipeline ──────────────────────────────────
Input (10 features)
  → RobustScaler → PolynomialFeatures(deg=2)
  → 65 features

Stage 1 — Classifier (มีหรือไม่มีการฟอกขาว?)
  Input(65) → Dense(64/32) → Sigmoid
  threshold = 0.48 (loaded from threshold_GBE.pkl)
               │
               ▼ ถ้า > threshold → ส่งต่อ Stage 2
Stage 2 — Regressor (ทำนายเปอร์เซ็นต์)
  Input(65) → Dense(128/64/32/16) → Linear
  Target: sqrt(% bleaching)  →  square() → 0–100%
                """,
                language="text",
            )
            st.markdown("""
| รายการ | รายละเอียด |
|---|---|
| **ชุดข้อมูล** | Global Bleaching Events (GBE) |
| **Task** | 2-Stage Regression |
| **Output** | เปอร์เซ็นต์การฟอกขาว (0–100%) |
| **Stage 1 Loss** | Binary Crossentropy |
| **Stage 2 Loss** | Huber |
| **Optimizer** | Adam |
| **Scaler** | RobustScaler + PolynomialFeatures(deg=2) |
""")

        st.markdown("**Features ที่ใช้ (ROC NN — 6 ฟีเจอร์):**")
        st.markdown("""
| Feature | คำอธิบาย |
|---|---|
| `Latitude` | ละติจูด |
| `Longitude` | ลองจิจูด |
| `SST (°C)` | อุณหภูมิผิวน้ำทะเล |
| `pH Level` | ระดับความกรดด่าง |
| `Abs_Latitude` | ค่าสัมบูรณ์ของละติจูด (Derived) |
| `SST_pH_Stress` | SST / pH — ดัชนีความเครียดความร้อน (Derived) |
""")

        st.markdown("**Features ที่ใช้ (GBE NN — 10 ฟีเจอร์ → 65 หลัง Polynomial):**")
        st.markdown("""
| Feature | คำอธิบาย |
|---|---|
| `Depth_m` | ความลึก (เมตร) |
| `Distance_to_Shore` | ระยะห่างจากชายฝั่ง |
| `Turbidity` | ความขุ่นของน้ำ |
| `Windspeed` | ความเร็วลม (m/s) |
| `Cyclone_Frequency` | ความถี่การเกิดพายุหมุน |
| `Temperature_Maximum` | อุณหภูมิสูงสุด (Kelvin) |
| `SSTA` | Sea Surface Temperature Anomaly |
| `SSTA_Maximum` | ค่า SSTA สูงสุด |
| `SSTA_DHW` | ค่าความร้อนสะสม (Degree Heating Weeks) |
| `TSA_DHW` | Thermal Stress Anomaly DHW |
""")

        st.divider()
        st.subheader("ผลการเทรนโมเดล")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown("**ROC — Training & Validation Curves**")
            try:
                st.image("models/ann_roc_training_curves.png", use_container_width=True)
            except Exception:
                st.info("ไม่พบไฟล์ภาพ `models/ann_roc_training_curves.png`")
        with img_col2:
            st.markdown("**GBE — Actual vs Predicted**")
            try:
                st.image("models/ann_gbe_actual_vs_predicted.png", use_container_width=True)
            except Exception:
                st.info("ไม่พบไฟล์ภาพ `models/ann_gbe_actual_vs_predicted.png`")

    # ─── Summary Table ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("สรุปเปรียบเทียบโมเดลทั้งหมด")
    st.markdown("""
| โมเดล | ชุดข้อมูล | Task | เทคนิค |
|---|---|---|---|
| **Stacking Ensemble** | ROC | Multiclass Classification (3 class) | RF + SVM + KNN → LogReg |
| **Stacking Ensemble** | GBE | Regression | RF + SVR + KNN → LinearReg |
| **Neural Network** | ROC | Binary Classification | ANN (Dense + Dropout) |
| **Neural Network** | GBE | 2-Stage Regression | Stage1 ANN Classifier + Stage2 ANN Regressor |
""")
