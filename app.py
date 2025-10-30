import os
import cv2
import json
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from ultralytics import YOLO
from crewai import Agent, Task, Crew
from openai import OpenAI

# =========================================================
# 🔹 Load environment variables
# =========================================================
load_dotenv("api.env")

# 🔹 Model path
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    r"C:\Desktop\vision_agent\runs\detect\car_detector_v2\weights\best.pt"
)

DEFAULT_IMG_PATH = r"C:\Desktop\vision_agent\data\raw\Cars Detection\valid\images\4c40c429a5a070e8_jpg.rf.L1Ey33Unmsn2ItPAAJFF.jpg"

# 🔹 Initialize YOLO
model = YOLO(MODEL_PATH)

# =========================================================
# 🔹 Streamlit UI Configuration
# =========================================================
st.set_page_config(
    page_title="🚗 Vehicle Vision AI",
    layout="wide"
)

# =========================================================
# 🔹 Sidebar
# =========================================================
with st.sidebar:
    st.image("./assets/ultra.svg", width=250)

    Openai_key = st.text_input(
        "OpenAI API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password"
    )

    labellerr_key = st.text_input(
        "LABELLERR_CLIENT_ID",
        value=os.getenv("LABELLERR_CLIENT_ID", ""),
        type="password"
    )

    if st.button("💾 Save Keys"):
        st.session_state["OPENAI_API_KEY"] = Openai_key
        st.session_state["LABELLERR_CLIENT_ID"] = labellerr_key

    if Openai_key or labellerr_key:
        st.success("Keys saved for this session")

    with st.expander("Confidence Threshold", expanded=True):
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            help="Adjust the minimum confidence level for detections"
        )

    with st.expander("Display Settings", expanded=True):
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        resize_option = st.checkbox("Resize Output Image", value=True)

    st.markdown("---")

    with st.expander("### 📊 About", expanded=True):
        st.markdown("""
        This AI system detects vehicles using YOLOv8 object detection.

        **Features:**
        - Real-time object detection
        - Multiple vehicle types
        - Confidence scoring
        - Export capabilities
        """)
        st.markdown("---")
        st.markdown("*Built with Streamlit & Ultralytics YOLO*")

# =========================================================
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #FAFAFA; }
.main-header { font-size: 2.8rem; color: #00D4B1; text-align: center; font-weight: 800; margin-bottom: 0.2rem; letter-spacing: -0.5px; }
.sub-header { text-align: center; color: #888; margin-bottom: 2.5rem; font-size: 1.1rem; }
.stButton>button { width: 100%; background-color: #555555; color: #FAFAFA; border: none; padding: 0.8rem 1rem; border-radius: 8px; font-weight: 600; font-size: 1rem; margin-top: 1rem; }
.stButton>button:hover { background-color: #777777; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
.stTextInput>div>div>input { background-color: #262730; color: #FAFAFA; border: 1px solid #393946; border-radius: 8px; padding: 0.8rem; }
.stTextInput>div>div>input:focus { border-color: #00D4B1; box-shadow: 0 0 0 2px rgba(0, 212, 177, 0.2); }
.css-1d391kg, .css-1d391kg>div { background-color: #0E1117 !important; border-right: 1px solid #262730; }
.css-1d391kg h1,h2,h3,h4,h5,h6,p,label { color: #FAFAFA !important; }
.stProgress > div > div > div > div { background-color: #00D4B1; }
.streamlit-expanderHeader { background-color: #262730; color: #FAFAFA; border-radius: 8px; font-weight: 600; }
.streamlit-expanderContent { background-color: #1A1D25; border-radius: 0 0 8px 8px; }
.card { background-color: #262730; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; border-left: 4px solid #00D4B1; }
.stRadio > div { background-color: #262730; padding: 1rem; border-radius: 8px; }
label { font-weight: 600 !important; margin-bottom: 0.5rem; display: block; color: #CCC !important; }
.main-title { font-size: 40px; font-weight: bold; display: flex; align-items: center; }
.main-title img { height: 70px; margin-left: 20px; vertical-align: middle; }
.subtitle { font-size: 20px; color: #AAAAAA; }
</style>
<div class="header">
  <div class="main-title">
    AutoVisionX
    <img src="https://cdn.prod.website-files.com/62f35fc537dc73303f60c5dc/6719f4a455439e70f5a15da3_labellerr-labelling%20made%20easy.webp" alt="Labellerr-ai Logo">
    <span style="margin-left:40px;">&</span>
    <img src="https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/crewai-brand-color.png" alt="crewai Logo">
  </div>
  <div class="subtitle"> Intelligent Vehicle Scene Understanding</div>
  <br>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader(
        "Upload an image for detection",
        type=["jpg", "jpeg", "png"],
        help="Upload vehicle images for AI analysis"
    )

if uploaded is not None:
    st.success(f"✅ Using uploaded image: `{uploaded.name}`")
    img = Image.open(uploaded).convert("RGB")
else:
    st.info(f"🖼 No image uploaded — using dataset image: `{os.path.basename(DEFAULT_IMG_PATH)}`")
    img = Image.open(DEFAULT_IMG_PATH).convert("RGB")

img_array = np.array(img)

# =========================================================
# 🔹 Object Detection
# =========================================================
with st.spinner("🔍 Running object detection... Please wait"):
    results = model(img_array, conf=confidence_threshold)
    res_img = results[0].plot()

if resize_option:
    res_img = cv2.resize(res_img, (600, 400))

col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Original Image")
    display_img = cv2.resize(np.array(img), (600, 400)) if resize_option else np.array(img)
    st.image(display_img, use_container_width=False)

with col2:
    st.subheader("🎯 Detection Results")
    st.image(res_img, use_container_width=False)

# =========================================================
# 🔹 Detection Summary
# =========================================================
st.markdown("---")
st.subheader("📊 Detection Analytics")

detections_json = results[0].to_json()
detections = json.loads(detections_json)

counts = {}
confidence_scores = {}

for obj in detections:
    name = obj.get("name", "unknown")
    counts[name] = counts.get(name, 0) + 1

    if name not in confidence_scores:
        confidence_scores[name] = []
    confidence_scores[name].append(obj.get("confidence", 0))

if counts:
    total_objects = sum(counts.values())

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Total Objects Detected", total_objects)
    with metric_col2:
        st.metric("Unique Classes", len(counts))
    with metric_col3:
        avg_confidence = np.mean([np.mean(scores) for scores in confidence_scores.values()])
        st.metric("Average Confidence", f"{avg_confidence:.2%}")

    st.markdown("#### 🎯 Detailed Breakdown")
    for cls, cnt in counts.items():
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown(f"**{cls.title()}**")
        with col2:
            st.markdown(f"**Count:** {cnt}")
        with col3:
            if show_confidence and cls in confidence_scores:
                avg_conf = np.mean(confidence_scores[cls])
                st.markdown(f"**Avg Confidence:** {avg_conf:.2%}")

    st.markdown("#### 📈 Distribution Chart")
    chart_data = {"Class": list(counts.keys()), "Count": list(counts.values())}
    st.bar_chart(chart_data, x="Class", y="Count")

else:
    st.warning("⚠️ No objects detected in this image. Try adjusting the confidence threshold or using a different image.")

# =========================================================
# 🔹 CrewAI Multi-Agent Analysis
# =========================================================
try:
    st.markdown("---")
    st.subheader("🤖 Agent Analysis")

    if not Openai_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to enable CrewAI summary.")
    else:
        client = OpenAI(api_key=Openai_key)

        vision_agent = Agent(
            role="Vision Analyst",
            goal="Summarize YOLO detection results into plain language.",
            backstory="Expert in analyzing what objects appear in a scene."
        )

        insight_agent = Agent(
            role="Environment Analyst",
            goal="Provide contextual insights about traffic, vehicles, or crowd density.",
            backstory="Understands city and transport scenes deeply."
        )

        report_agent = Agent(
            role="Report Compiler",
            goal="Combine the above results into a concise JSON report.",
            backstory="Formats analytical insights in structured JSON."
        )

        vision_task = Task(
            description=f"Detected objects: {counts}. Write a human-readable summary (2 lines).",
            expected_output="Short text summary.",
            agent=vision_agent
        )

        insight_task = Task(
            description="Based on the summary, generate 2-3 insights about traffic conditions.",
            expected_output="Short bullet points.",
            agent=insight_agent
        )

        report_task = Task(
            description="Combine summary + insights into JSON with 'summary' and 'insights' keys.",
            expected_output="JSON formatted output.",
            agent=report_agent
        )

        crew = Crew(
            agents=[vision_agent, insight_agent, report_agent],
            tasks=[vision_task, insight_task, report_task]
        )

        # Run CrewAI
        result = crew.kickoff(inputs={"counts": counts})

        # 🛠 Safe result extraction
        output_data = None
        if isinstance(result, dict):
            output_data = result.get("results") or result.get("output") or result
        else:
            output_data = result

        st.markdown("### 🧠 Scene Summary")

        try:
            if isinstance(output_data, dict):
                summary = output_data.get("summary", "No summary found.")
                insights = output_data.get("insights", [])

                st.write(summary)

                st.markdown("### 💡 Analytical Insights")
                if insights:
                    for i, insight in enumerate(insights, start=1):
                        st.markdown(f"**{i}.** {insight}")
                else:
                    st.info("No insights found in CrewAI output.")

                st.markdown("### 📋 JSON Report")
                st.json(output_data)

            elif isinstance(output_data, list):
                st.write(output_data[0].get("output", "No summary found."))

                st.markdown("### 💡 Analytical Insights")
                if len(output_data) > 1:
                    st.write(output_data[1].get("output", "No insights found."))
                else:
                    st.info("No insights found in CrewAI output.")

                st.markdown("### 📋 JSON Report")
                if len(output_data) > 2:
                    try:
                        st.json(json.loads(output_data[2].get("output", "{}")))
                    except Exception:
                        st.text(output_data[2].get("output", "{}"))
                else:
                    st.info("JSON report not found in CrewAI output.")

            else:
                try:
                    parsed = json.loads(str(output_data))
                    st.json(parsed)
                except Exception:
                    st.write(str(output_data))

        except Exception as e:
            st.error(f"⚠️ Output parsing error: {e}")

except Exception as e:
    st.error(f"❌ CrewAI Error: {e}")

# =========================================================
# 🔹 Export Options
# =========================================================
st.markdown("---")
st.subheader("📥 Export Results")

exp_col1, exp_col2, exp_col3 = st.columns(3)

with exp_col1:
    if st.button("💾 Save Detection Image"):
        _, buffer = cv2.imencode('.jpg', res_img)
        st.download_button(
            label="Download Detection Image",
            data=buffer.tobytes(),
            file_name="detection_result.jpg",
            mime="image/jpeg"
        )

with exp_col2:
    if st.button("📊 Export Statistics"):
        stats = {
            "total_objects": total_objects if counts else 0,
            "class_distribution": counts,
            "confidence_scores": {k: [float(score) for score in v] for k, v in confidence_scores.items()}
        }
        st.download_button(
            label="Download JSON Report",
            data=json.dumps(stats, indent=2),
            file_name="detection_report.json",
            mime="application/json"
        )

with exp_col3:
    if st.button("🔄 Reset Session"):
        st.rerun()

# =========================================================
# 🔹 Custom CSS Styling
# =========================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
    }
    .css-1d391kg {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)
