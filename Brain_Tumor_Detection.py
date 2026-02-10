import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go

# ---------------------------
# BASIC CONFIG & PAGE SETUP
# ---------------------------
st.set_page_config(
    page_title="Brain Tumour Classifier",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------
# CUSTOM CSS
# ---------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #eef2ff 0%, #f9fafb 40%, #fefce8 100%);
        color: #0f172a;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main > div {
        padding-top: 1.5rem;
    }

    /* HEADER */
    .app-header {
        padding: 1.5rem 1.75rem;
        border-radius: 1.25rem;
        background: radial-gradient(circle at top left, #4f46e5 0, #1f2937 55%, #020617 100%);
        color: #e5e7eb;
        box-shadow: 0 18px 40px rgba(15,23,42,0.45);
        margin-bottom: 1.75rem;
    }
    .app-header-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .app-header-subtitle {
        font-size: 0.98rem;
        color: #e5e7eb;
        opacity: 0.9;
    }
    .app-header-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.8rem;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: rgba(15,23,42,0.5);
        margin-top: 0.75rem;
    }

    /* CARDS */
    .card {
        background-color: #ffffff;
        padding: 1.5rem 1.6rem;
        border-radius: 1.25rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 14px 30px rgba(15,23,42,0.08);
    }

    .card-soft {
        background-color: rgba(255,255,255,0.92);
        padding: 1.4rem 1.5rem;
        border-radius: 1.1rem;
        border: 1px solid rgba(226, 232, 240, 0.9);
        backdrop-filter: blur(6px);
        box-shadow: 0 10px 25px rgba(15,23,42,0.06);
    }

    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.6rem;
    }

    .helper-text {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.8rem;
    }

    /* METRICS */
    .metrics-row {
        display: flex;
        gap: 1.1rem;
        margin-top: 0.75rem;
        margin-bottom: 0.5rem;
        flex-wrap: wrap;
    }

    .metric-pill {
        flex: 1;
        min-width: 140px;
        padding: 0.75rem 0.9rem;
        border-radius: 0.9rem;
        border: 1px solid #e5e7eb;
        background: #f9fafb;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6b7280;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 600;
        color: #111827;
    }

    .prob-header {
        font-size: 0.9rem;
        font-weight: 600;
        color: #374151;
        margin-top: 0.8rem;
        margin-bottom: 0.3rem;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    section[data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] strong,
    section[data-testid="stSidebar"] .element-container {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .sidebar-muted {
        color: #cbd5e1 !important;
        font-size: 0.86rem;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #475569 !important;
    }
    section[data-testid="stSidebar"] .caption,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
    }

    /* FOOTER */
    .footer-text {
        font-size: 0.78rem;
        color: #6b7280;
        text-align: center;
        margin-top: 2.2rem;
        margin-bottom: 0.5rem;
    }

    /* High-quality image rendering */
    img {
        image-rendering: -webkit-optimize-contrast;
        image-rendering: high-quality;
        image-rendering: crisp-edges;
    }

    /* PROCEDURE CONTAINER */
    .procedure-box{
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(226,232,240,0.95);
        border-radius: 1.1rem;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 12px 28px rgba(15,23,42,0.08);
        backdrop-filter: blur(6px);
    }
    .procedure-box b, .procedure-box strong{
        font-weight: 800;
    }
    .procedure-box p{
        margin: 0;
        color: #111827;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# CONSTANTS & LOADERS
# ---------------------------
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model_and_labels():
    model = keras.models.load_model("brain_tumor_final_model.h5")
    with open("class_labels.json", "r") as f:
        class_labels = json.load(f)
    return model, class_labels

model, class_labels = load_model_and_labels()

# ---------------------------
# PROCEDURE MAP (BOLD + HTML)
# ---------------------------
procedure_map = {
    "glioma": """
    <b>Consult a neurosurgeon.</b><br>
    <b>Next steps:</b> biopsy may be considered, surgical resection when appropriate, and
    radiotherapy/chemotherapy depending on tumour grade.
    """,

    "meningioma": """
    <b>Surgical resection</b> if accessible.<br>
    <b>Follow-up:</b> regular MRI monitoring.<br>
    <b>Alternative:</b> radiotherapy for residual or inoperable cases.
    """,

    "notumor": """
    <b>No tumour detected.</b><br>
    <b>Advice:</b> correlate with symptoms and consider further imaging or specialist review
    if concerns persist.
    """,

    "pituitary": """
    <b>Endocrine assessment.</b><br>
    <b>Options:</b> transsphenoidal surgery may be considered; medical therapy or radiotherapy
    depending on tumour type.
    """
}

# ---------------------------
# MRI-LIKE CHECK
# ---------------------------
def is_brain_mri_like_pil(pil_img: Image.Image, color_threshold: float = 10.0) -> bool:
    img_rgb = pil_img.convert("RGB")
    arr = np.array(img_rgb).astype("float32")

    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]

    diff_rg = np.mean(np.abs(r - g))
    diff_rb = np.mean(np.abs(r - b))
    diff_gb = np.mean(np.abs(g - b))

    color_diff = (diff_rg + diff_rb + diff_gb) / 3.0
    return color_diff < color_threshold

# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr)[0]
    idx = int(np.argmax(probs))
    pred_class = class_labels[idx]
    confidence = float(probs[idx])
    procedure = procedure_map.get(pred_class, "<b>Consult specialist.</b>")
    return pred_class, confidence, procedure, probs

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.markdown("### 🧠 Brain Tumour Classifier")
    st.markdown(
        """
        <p class="sidebar-muted">
        This prototype uses a trained CNN to classify a single MRI slice
        into one of four tumour categories.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(
        """
        **How to use**

        1. Upload a brain MRI image (`.png`, `.jpg`, `.jpeg`)
        2. Click **Run prediction**
        3. Review the predicted class and confidence

        **Classes**
        - Glioma  
        - Meningioma  
        - No tumour  
        - Pituitary tumour
        """
    )

    st.markdown("---")
    st.markdown("**Model status:** ✅ Loaded")
    st.caption("Educational use only. Not for clinical decision-making.")

# ---------------------------
# HEADER
# ---------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="app-header-title">Brain Tumour MRI Classification</div>
        <div class="app-header-subtitle">
            Deep learning–based coursework prototype that screens a single MRI slice for 
            four brain tumour categories. Designed for educational and research purposes.
        </div>
        <div class="app-header-badge">
            <span>⚗️ CNN model</span>
            <span>•</span>
            <span>4-way classification</span>
            <span>•</span>
            <span>Non-clinical demo</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# MAIN LAYOUT
# ---------------------------
col_left, col_right = st.columns([1.1, 1])

# LEFT: upload & preview
with col_left:
    upload_container = st.container()
    with upload_container:
        st.markdown('<div class="card-title">📤 Upload MRI slice</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="helper-text">Upload a single axial brain MRI slice. '
            'The model expects a grayscale-like image; colourful photos will be rejected.</div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Upload MRI image",
            type=["png", "jpg", "jpeg"]
        )

        img_display = None
        if uploaded_file is not None:
            try:
                img_display = Image.open(uploaded_file)

                # Convert to high-quality PNG bytes for better display quality
                from io import BytesIO

                # Ensure image is in RGB mode
                if img_display.mode != 'RGB':
                    img_display = img_display.convert('RGB')

                # Create high-quality PNG buffer
                buffer = BytesIO()
                img_display.save(buffer, format='PNG', optimize=False, compress_level=0)
                buffer.seek(0)

                # Display with maximum quality
                st.image(buffer, caption="Uploaded MRI preview", use_container_width=True)
            except Exception as e:
                st.error(f"Could not read image: {e}")
                img_display = None

        st.markdown("</div>", unsafe_allow_html=True)

# RIGHT: prediction
with col_right:
    pred_container = st.container()
    with pred_container:
        st.markdown('<div class="card-title">Prediction panel</div>', unsafe_allow_html=True)

        if uploaded_file is None or img_display is None:
            st.info("Upload an MRI image on the left to enable prediction.")
        else:
            run_pred = st.button("🔍Run prediction", use_container_width=True)

            if run_pred:
                if not is_brain_mri_like_pil(img_display):
                    st.error(
                        "The uploaded image does not appear to be a brain MRI scan. "
                        "Please upload a valid MRI slice (typically grayscale-looking)."
                    )
                else:
                    pred_class, conf, procedure, probs = predict_image(img_display)

                    st.markdown('<div class="metrics-row">', unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(
                            f"""
                            <div class="metric-pill">
                                <div class="metric-label">Predicted class</div>
                                <div class="metric-value">{pred_class.title()}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with m2:
                        st.markdown(
                            f"""
                            <div class="metric-pill">
                                <div class="metric-label">Confidence</div>
                                <div class="metric-value">{conf * 100:.1f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)

                    tabs = st.tabs(["📊 Class probabilities", "📝 Suggested procedure"])

                    with tabs[0]:
                        st.markdown('<div class="prob-header">Model output distribution</div>', unsafe_allow_html=True)

                        prob_dict = {label: float(p) * 100 for label, p in zip(class_labels, probs)}
                        sorted_items = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                        labels = [item[0].title() for item in sorted_items]
                        values = [item[1] for item in sorted_items]

                        fig = go.Figure(data=[
                            go.Bar(
                                x=values,
                                y=labels,
                                orientation='h',
                                marker=dict(
                                    color=values,
                                    colorscale='Viridis',
                                    showscale=False,
                                    line=dict(color='rgba(0,0,0,0.2)', width=1)
                                ),
                                text=[f'{v:.1f}%' for v in values],
                                textposition='auto',
                                textfont=dict(size=14, color='white', family='Arial Black'),
                                hovertemplate='<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>'
                            )
                        ])

                        fig.update_layout(
                            height=400,
                            margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(249,250,251,0.5)',
                            xaxis=dict(
                                title='Probability (%)',
                                titlefont=dict(size=14, color='#374151'),
                                tickfont=dict(size=12, color='#6b7280'),
                                gridcolor='rgba(156,163,175,0.2)',
                                showgrid=True,
                                range=[0, 100]
                            ),
                            yaxis=dict(
                                titlefont=dict(size=14, color='#374151'),
                                tickfont=dict(size=13, color='#111827', family='Arial'),
                                showgrid=False
                            ),
                            font=dict(family='Arial', size=12),
                            hoverlabel=dict(
                                bgcolor="white",
                                font_size=13,
                                font_family="Arial"
                            )
                        )

                        st.plotly_chart(fig, use_container_width=True, config={
                            'displayModeBar': False,
                            'staticPlot': False,
                            'toImageButtonOptions': {
                                'format': 'png',
                                'width': 1920,
                                'height': 1080,
                                'scale': 3
                            }
                        })

                    with tabs[1]:
                        # Render the procedure_map HTML inside a styled container
                        st.markdown(
                            f"""
                            <div class="procedure-box">
                                <p>{procedure}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.warning(
                            "This is a research/educational prototype and **not** a medical device. "
                            "No diagnosis or treatment decisions should be made from this output."
                        )

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# ABOUT SECTION
# ---------------------------
with st.expander("ℹ️ About this app"):
    st.markdown(
        """
        This application demonstrates a convolutional neural network (CNN) trained on
        a brain tumour MRI dataset with four classes:
        **glioma**, **meningioma**, **no tumour**, and **pituitary tumour**.

        **Pipeline summary**
        - Data cleaning (removal of corrupted images and near-duplicates)
        - Train/validation/test split via directory structure
        - Data augmentation (rotations, flips, small translations)
        - Baseline CNN → deeper CNN → 5-conv CNN comparison
        - Final deployment uses the model that achieved the best test accuracy

        A pre-check rejects non–MRI-like images (e.g. colourful natural photos)
        to reduce misuse on irrelevant inputs.

        The suggested procedure text is illustrative and does **not** constitute
        clinical advice. Any real diagnosis or treatment decisions must be made by
        qualified healthcare professionals.
        """
    )

st.markdown(
    '<div class="footer-text">'
    'Brain Tumour Classification • Deep Learning • Built with Streamlit'
    '</div>',
    unsafe_allow_html=True
)
