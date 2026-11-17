
import streamlit as st
import pandas as pd
import plotly.express as px

from src.utils import setup_page, page_header, render_disclaimer

setup_page("Model Performance", "📊")
page_header("Model Performance", "YOLOv8-Nano evaluation on the test dataset.")

st.markdown("""
<style>
.metric-card {
    background: white;
    border: 1px solid #dce5f0;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 12px;
}
.metric-label { color: #52647a; font-size: 14px; }
.metric-value { color: #0875d1; font-size: 28px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown("### Overall Performance")

metrics = [
    ("Precision", "32.0%"),
    ("Recall", "28.1%"),
    ("mAP@50", "20.5%"),
    ("mAP@50-95", "5.6%"),
    ("Test Images", "621"),
    ("Ground Truth Boxes", "347")
]

cols = st.columns(3)

for i, (label, value) in enumerate(metrics):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.caption("Evaluation: confidence=0.001 | IoU=0.60 | Image size=640")

st.markdown("### Class-wise Detection Performance")

df = pd.DataFrame({
    "Class": [
        "Elbow", "Fingers", "Forearm",
        "Humerus", "Shoulder", "Wrist"
    ],
    "mAP@50 (%)": [7.93, 12.7, 31.8, 31.4, 15.3, 24.2]
})

fig = px.bar(
    df,
    x="Class",
    y="mAP@50 (%)",
    text="mAP@50 (%)",
    title="Class-wise mAP@50"
)

fig.update_traces(
    textfont_color="#17324d",
    textposition="outside"
)

fig.update_layout(
    font=dict(color="#17324d", size=13),
    title_font_color="#17324d",
    plot_bgcolor="white",
    paper_bgcolor="white",
    yaxis=dict(
        range=[0, 40],
        title_font_color="#17324d",
        tickfont_color="#17324d"
    ),
    xaxis=dict(tickfont_color="#17324d")
)

st.plotly_chart(fig, width="stretch")

st.info(
    "This is an experimental research prototype. "
    "It is not clinically validated for medical diagnosis."
)

render_disclaimer()
