"""
Streamlit Frontend — SentinAir: Air Quality Classification
"""

import os
import streamlit as st
import requests
import pandas as pd

API_URL  = os.environ.get("API_URL", "http://localhost:8000")
LABEL_MAP = {0: "🟢 Good", 1: "🟡 Moderate", 2: "🔴 Poor"}

st.set_page_config(page_title="SentinAir", page_icon="🌬️", layout="wide")
st.title("🌬️ SentinAir: Air Quality Classification")
st.caption("Real-time air quality classification using multivariate sensor data.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        if health.status_code == 200:
            st.success("✅ API Online")
        else:
            st.error(f"⚠️ API returned {health.status_code}: {health.json().get('detail','')}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach API")

    st.markdown("---")
    st.markdown("**Label Guide**")
    for k, v in LABEL_MAP.items():
        st.markdown(f"- `{k}` → {v}")

    st.markdown("---")
    if st.button("🔄 Check Feedback Stats"):
        try:
            r = requests.get(f"{API_URL}/feedback/stats", timeout=5)
            if r.status_code == 200:
                stats = r.json()
                st.metric("Total Feedback Rows", stats['total_feedback_rows'])
                st.metric("Error Rate", f"{stats['error_rate']*100:.1f}%")
                st.json(stats['source_breakdown'])
            else:
                st.warning("Could not fetch stats")
        except Exception as e:
            st.error(str(e))

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📡 Single Prediction", "📂 Batch Prediction", "📊 Feedback Monitor"])

# ── Tab 1: Single prediction ───────────────────────────────────────────────────
with tab1:
    st.subheader("Enter Sensor Readings")
    st.info("All sensor values are numeric. Leave as default to try a sample reading.")

    col1, col2, col3 = st.columns(3)
    with col1:
        pt08_s1 = st.number_input("PT08.S1(CO)",    value=1200.0)
        c6h6    = st.number_input("C6H6(GT)",        value=5.0)
        pt08_s2 = st.number_input("PT08.S2(NMHC)",  value=900.0)
        nox     = st.number_input("NOx(GT)",          value=100.0)
    with col2:
        pt08_s3 = st.number_input("PT08.S3(NOx)",   value=1000.0)
        no2     = st.number_input("NO2(GT)",          value=80.0)
        pt08_s4 = st.number_input("PT08.S4(NO2)",   value=1500.0)
        pt08_s5 = st.number_input("PT08.S5(O3)",    value=1000.0)
    with col3:
        temp = st.number_input("Temperature (T)",        value=15.0)
        rh   = st.number_input("Relative Humidity (RH)", value=50.0)
        ah   = st.number_input("Absolute Humidity (AH)", value=0.75)

    features = {
        "PT08.S1(CO)": pt08_s1, "C6H6(GT)": c6h6, "PT08.S2(NMHC)": pt08_s2,
        "NOx(GT)": nox, "PT08.S3(NOx)": pt08_s3, "NO2(GT)": no2,
        "PT08.S4(NO2)": pt08_s4, "PT08.S5(O3)": pt08_s5,
        "T": temp, "RH": rh, "AH": ah,
    }

    if st.button("🔍 Predict Air Quality", use_container_width=True):
        with st.spinner("Running inference..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=10)
                if resp.status_code == 200:
                    result = resp.json()
                    pred_label = LABEL_MAP.get(result['prediction'], str(result['prediction']))
                    st.success(f"### Predicted: {pred_label}")
                    prob_df = pd.DataFrame({
                        "Class": [LABEL_MAP.get(i, str(i)) for i in range(len(result['probability']))],
                        "Probability": [round(p, 4) for p in result['probability']],
                    })
                    st.bar_chart(prob_df.set_index("Class"))
                    st.session_state['last_prediction'] = result['prediction']
                    st.session_state['last_features']   = features
                else:
                    st.error(f"API Error {resp.status_code}: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not reach the API.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    if 'last_prediction' in st.session_state:
        st.markdown("---")
        st.subheader("📝 Provide Feedback")
        actual_label = st.selectbox(
            "What was the actual air quality?",
            options=list(LABEL_MAP.keys()),
            format_func=lambda x: LABEL_MAP[x],
        )
        if st.button("Submit Feedback"):
            fb_data = {
                "prediction": st.session_state['last_prediction'],
                "actual": actual_label,
                "features": st.session_state['last_features'],
                "source": "single",
            }
            try:
                fb_resp = requests.post(f"{API_URL}/feedback", json=fb_data, timeout=10)
                if fb_resp.status_code == 200:
                    st.success(f"✅ Feedback stored! (id={fb_resp.json().get('id')})")
                    del st.session_state['last_prediction']
                    del st.session_state['last_features']
                else:
                    st.error(f"Feedback failed: {fb_resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

# ── Tab 2: Batch prediction ────────────────────────────────────────────────────
with tab2:
    st.subheader("Batch Prediction via CSV Upload")
    st.markdown(
        "Upload a CSV with the 11 sensor columns. "
        "If a `Target` column exists it is used to auto-submit feedback after prediction."
    )
    st.code("PT08.S1(CO), C6H6(GT), PT08.S2(NMHC), NOx(GT), PT08.S3(NOx), "
            "NO2(GT), PT08.S4(NO2), PT08.S5(O3), T, RH, AH", language="text")

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write(f"**Loaded {len(df)} rows.** Preview:")
        st.dataframe(df.head())

        has_labels = 'Target' in df.columns

        col_a, col_b = st.columns(2)
        run_pred   = col_a.button("▶️ Run Batch Prediction", use_container_width=True)
        submit_fb  = col_b.checkbox(
            "Auto-submit feedback after prediction",
            value=has_labels,
            disabled=not has_labels,
            help="Requires a 'Target' column with class names (Good/Moderate/Poor)",
        )

        if run_pred:
            FEATURE_COLS = ["PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)",
                            "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
                            "T", "RH", "AH"]
            label_to_int = {"Good": 0, "Moderate": 1, "Poor": 2}

            missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
            if missing_cols:
                st.error(f"CSV is missing columns: {missing_cols}")
            else:
                rows_payload = [
                    {c: float(row[c]) for c in FEATURE_COLS}
                    for _, row in df.iterrows()
                ]

                with st.spinner(f"Running inference on {len(df)} rows..."):
                    try:
                        batch_resp = requests.post(
                            f"{API_URL}/predict/batch",
                            json={"rows": rows_payload},
                            timeout=60,
                        )
                        if batch_resp.status_code != 200:
                            st.error(f"Batch API error: {batch_resp.text}")
                        else:
                            batch_results = batch_resp.json()['results']
                            preds  = [r.get('prediction') for r in batch_results]
                            labels = [r.get('label', '') for r in batch_results]

                            df['Predicted_Class'] = preds
                            df['Predicted_Label'] = labels
                            st.success(f"Done! {sum(p is not None for p in preds)}/{len(df)} rows predicted.")
                            st.dataframe(df)

                            # ── Auto feedback submission ──────────────────
                            if submit_fb and has_labels:
                                fb_items = []
                                for i, (_, row) in enumerate(df.iterrows()):
                                    pred_int = preds[i]
                                    if pred_int is None:
                                        continue
                                    raw_label = str(row['Target']).strip()
                                    actual_int = label_to_int.get(raw_label)
                                    if actual_int is None:
                                        continue
                                    fb_items.append({
                                        "prediction": pred_int,
                                        "actual": actual_int,
                                        "features": {c: float(row[c]) for c in FEATURE_COLS},
                                        "source": "batch",
                                    })

                                if fb_items:
                                    fb_resp = requests.post(
                                        f"{API_URL}/feedback/batch",
                                        json={"items": fb_items},
                                        timeout=30,
                                    )
                                    if fb_resp.status_code == 200:
                                        st.success(
                                            f"✅ {len(fb_items)} feedback rows stored automatically."
                                        )
                                    else:
                                        st.warning(f"Feedback storage failed: {fb_resp.text}")
                                else:
                                    st.warning("No valid feedback items to store (check Target column values).")

                            csv_out = df.to_csv(index=False).encode()
                            st.download_button(
                                "⬇️ Download Results CSV",
                                data=csv_out,
                                file_name="sentinair_batch_results.csv",
                                mime="text/csv",
                            )
                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")

# ── Tab 3: Feedback Monitor ────────────────────────────────────────────────────
with tab3:
    st.subheader("📊 Feedback Collection Monitor")
    st.markdown("Use this tab to verify that feedback is being collected correctly.")

    if st.button("🔃 Refresh Stats"):
        try:
            r = requests.get(f"{API_URL}/feedback/stats", timeout=5)
            if r.status_code == 200:
                stats = r.json()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Feedback Rows", stats['total_feedback_rows'])
                c2.metric("Error Rate", f"{stats['error_rate']*100:.1f}%")
                c3.metric("Retraining Trigger?", "⚠️ YES" if stats['error_rate'] > 0.05 else "✅ No")

                st.markdown("**Feedback by source:**")
                st.json(stats['source_breakdown'])

                st.markdown("**Feedback by true class:**")
                st.json(stats['class_breakdown'])

                st.markdown("**Last 10 feedback records:**")
                if stats['recent_10']:
                    st.dataframe(pd.DataFrame(stats['recent_10']))
                else:
                    st.info("No feedback records yet.")
            else:
                st.error("Could not fetch stats")
        except Exception as e:
            st.error(str(e))