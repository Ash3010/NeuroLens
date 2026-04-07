import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights, VGG16_Weights
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import io, json

from model import get_model
from cam import run_cam
from utils import draw_boxes
from resnet_viz import run_resnet, get_resnet, preprocess as resnet_preprocess, extract_feature_maps as resnet_extract
from vgg_viz import run_vgg, get_vgg, preprocess as vgg_preprocess, extract_feature_maps as vgg_extract
from advanced_viz import make_pca_tsne, make_isosurface, make_variance_map
from eval_utils import prediction_metrics, cam_focus_score, feature_metrics, save_feedback_row, timestamp_now
st.set_page_config(page_title="NeuroLens · CNN Visualizer", page_icon="🧠", layout="wide")

IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

@st.cache_data
def load_imagenet_labels():
    import urllib.request
    try:
        with urllib.request.urlopen(IMAGENET_LABELS_URL) as r:
            return json.loads(r.read().decode())
    except:
        return [f"class_{i}" for i in range(1000)]

MODEL_INFO = {
    "ResNet50": {"params":"25.6M","layers":50,"depth":"4 residual blocks","input":"224×224","task":"Classification","special":"Skip connections prevent vanishing gradients","color":"#7c6bff"},
    "VGG16":    {"params":"138M","layers":16,"depth":"5 conv blocks","input":"224×224","task":"Classification","special":"Simple sequential — no skip connections","color":"#ff6b9d"},
    "YOLOv5":   {"params":"7.2M","layers":"~300","depth":"CSP backbone","input":"640×640","task":"Object Detection","special":"Single-pass detection — extremely fast","color":"#6bffd4"},
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{--bg:#0a0a0f;--surface:#12121a;--surface2:#1a1a26;--accent:#7c6bff;--accent2:#ff6b9d;--accent3:#6bffd4;--text:#e8e8f0;--muted:#6b6b80;--border:#2a2a3a}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg);color:var(--text)}
.stApp{background-color:var(--bg)}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:2rem 3rem;max-width:1400px}
.hero{text-align:center;padding:2.5rem 0 1.5rem}
.hero-title{font-family:'Space Mono',monospace;font-size:3rem;font-weight:700;background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 50%,var(--accent3) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0;letter-spacing:-1px}
.hero-sub{color:var(--muted);font-size:.85rem;margin-top:.4rem;letter-spacing:.1em;text-transform:uppercase}
.hero-line{width:60px;height:2px;background:linear-gradient(90deg,var(--accent),var(--accent2));margin:1rem auto 0;border-radius:2px}
.section-label{font-family:'Space Mono',monospace;font-size:.68rem;color:var(--accent);text-transform:uppercase;letter-spacing:.2em;margin-bottom:.8rem;display:flex;align-items:center;gap:.5rem}
.section-label::after{content:'';flex:1;height:1px;background:var(--border)}
.output-title{font-family:'Space Mono',monospace;font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.6rem}
.badge{display:inline-block;background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:.2rem .6rem;font-family:'Space Mono',monospace;font-size:.68rem;color:var(--accent3);margin-right:.3rem;margin-bottom:.3rem}
.info-box{background:var(--surface2);border-left:3px solid var(--accent);border-radius:0 8px 8px 0;padding:.7rem 1rem;font-size:.82rem;color:var(--muted);line-height:1.6;margin-bottom:.8rem}
.info-box strong{color:var(--text)}
.model-info-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1rem;margin-bottom:1rem}
.model-stat{display:flex;justify-content:space-between;padding:.25rem 0;border-bottom:1px solid var(--border);font-size:.8rem}
.model-stat:last-child{border-bottom:none}
.model-stat-key{color:var(--muted);font-family:'Space Mono',monospace;font-size:.7rem}
.pred-bar-wrap{margin-bottom:.4rem}
.pred-label{font-size:.78rem;color:var(--text);margin-bottom:.15rem;display:flex;justify-content:space-between}
.pred-bar-bg{background:var(--surface2);border-radius:4px;height:6px;overflow:hidden}
.pred-bar-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--accent),var(--accent2))}
.compare-header{font-family:'Space Mono',monospace;font-size:.8rem;color:var(--text);text-align:center;padding:.5rem;background:var(--surface2);border-radius:6px;margin-bottom:.5rem}
.stButton>button{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;color:white!important;border:none!important;border-radius:8px!important;font-family:'Space Mono',monospace!important;font-size:.78rem!important;width:100%}
div[data-testid="stImage"] img{border-radius:8px;border:1px solid var(--border)}
.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border-radius:8px;padding:.2rem;gap:.2rem}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;font-family:'Space Mono',monospace!important;font-size:.68rem!important;border-radius:6px!important}
.stTabs [aria-selected="true"]{background:var(--surface2)!important;color:var(--text)!important}
.stDownloadButton>button{background:var(--surface2)!important;color:var(--accent3)!important;border:1px solid var(--border)!important;border-radius:8px!important;font-family:'Space Mono',monospace!important;font-size:.72rem!important;width:100%}
hr{border-color:var(--border)!important;opacity:.4}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

def img_to_bytes(img_np):
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()

def get_top5(model, tensor, labels):
    with torch.no_grad():
        out = model(tensor)
    probs = torch.nn.functional.softmax(out[0], dim=0)
    top5_probs, top5_ids = torch.topk(probs, 5)
    return [(labels[i.item()], p.item()) for i, p in zip(top5_ids, top5_probs)]

def render_predictions(preds):
    html = '<div style="margin-bottom:1rem">'
    for label, prob in preds:
        pct = prob * 100
        html += f'<div class="pred-bar-wrap"><div class="pred-label"><span>{label.replace("_"," ").title()}</span><span style="color:var(--accent3);font-family:Space Mono,monospace">{pct:.1f}%</span></div><div class="pred-bar-bg"><div class="pred-bar-fill" style="width:{int(pct)}%"></div></div></div>'
    return html + '</div>'

def render_model_info(name):
    info = MODEL_INFO[name]
    c = info['color']
    return f'''<div class="model-info-card" style="border-top:2px solid {c}">
    <div style="font-family:Space Mono,monospace;font-size:.7rem;color:{c};text-transform:uppercase;letter-spacing:.15em;margin-bottom:.6rem">{info["task"]} Model</div>
    <div style="font-size:1.1rem;font-weight:600;color:var(--text);margin-bottom:.8rem">{name}</div>
    <div class="model-stat"><span class="model-stat-key">Parameters</span><span>{info["params"]}</span></div>
    <div class="model-stat"><span class="model-stat-key">Layers</span><span>{info["layers"]}</span></div>
    <div class="model-stat"><span class="model-stat-key">Architecture</span><span>{info["depth"]}</span></div>
    <div class="model-stat"><span class="model-stat-key">Input Size</span><span>{info["input"]}</span></div>
    <div style="margin-top:.6rem;font-size:.78rem;color:var(--muted);font-style:italic">{info["special"]}</div>
    </div>'''

def stamp_prediction(img_np, label, conf):
    img_pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)
    text = f"{label.replace('_',' ').title()} {conf*100:.0f}%"
    draw.rectangle([0,0,img_pil.width,28], fill=(0,0,0))
    draw.text((6,6), text, fill=(124,107,255))
    return np.array(img_pil)

def stat_card(val, label, color):
    return f'<div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:.8rem;text-align:center"><div style="font-family:Space Mono,monospace;font-size:1.3rem;color:{color}">{val}</div><div style="font-size:.75rem;color:#6b6b80">{label}</div></div>'

# ── Threshold slider (shared) ─────────────────────────────────────────────
def iso_threshold_slider(key_prefix):
    st.markdown('<div class="info-box">The scalar field is the <strong>mean activation</strong> averaged across all channels. Bright regions = where the network was most active overall. Use the sliders to set threshold levels — only regions <strong>above</strong> the threshold are highlighted (like an isosurface contour).</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    t1 = c1.slider("Threshold 1", 0.1, 0.9, 0.3, 0.05, key=f"{key_prefix}_t1")
    t2 = c2.slider("Threshold 2", 0.1, 0.9, 0.5, 0.05, key=f"{key_prefix}_t2")
    t3 = c3.slider("Threshold 3", 0.1, 0.9, 0.7, 0.05, key=f"{key_prefix}_t3")
    t4 = c4.slider("Threshold 4", 0.1, 0.9, 0.9, 0.05, key=f"{key_prefix}_t4")
    return sorted([t1, t2, t3, t4])

labels = load_imagenet_labels()

# ── Hero ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero"><p class="hero-sub">MM804 Graphics & Animation · Group Project</p><h1 class="hero-title">NeuroLens</h1><p class="hero-sub">Tensor-Based Visualization of Deep Neural Network Internals</p><div class="hero-line"></div></div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
    model_choice = st.radio("", ["🔵  ResNet50","🟣  VGG16","🟢  YOLOv5"], label_visibility="collapsed")
    model_name = "ResNet50" if "ResNet" in model_choice else "VGG16" if "VGG" in model_choice else "YOLOv5"
    st.markdown(render_model_info(model_name), unsafe_allow_html=True)

    st.markdown('<div class="section-label">Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"], label_visibility="collapsed")

    st.markdown('<div class="section-label">Options</div>', unsafe_allow_html=True)
    block_map = {"Block1":3,"Block2":8,"Block3":15,"Block4":22,"Block5":29}

    if "ResNet" in model_choice or "VGG" in model_choice:
        num_channels = st.slider("Channels to show", 4, 32, 16, 4)
    if "ResNet" in model_choice:
        layer_idx = st.slider("Layer to inspect (1-4)", 1, 4, 1)
        show_layers = [f"layer{layer_idx}", f"layer{min(layer_idx+2,4)}"]
    elif "VGG" in model_choice:
        block_idx = st.slider("Block to inspect (1-5)", 1, 5, 1)
        show_blocks = [(f"Block{block_idx}", block_map[f"Block{block_idx}"]),
                       (f"Block{min(block_idx+2,5)}", block_map[f"Block{min(block_idx+2,5)}"])]

    run_btn = st.button("▶  Run Visualization")

# ── No image ──────────────────────────────────────────────────────────────
if not uploaded_file:
    st.markdown('''<div style="text-align:center;padding:3rem 0;color:#6b6b80">
    <div style="font-size:2.5rem;margin-bottom:1rem">🧬</div>
    <div style="font-family:Space Mono,monospace;font-size:.78rem;letter-spacing:.1em;text-transform:uppercase">Upload an image in the sidebar to begin</div></div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:1rem">
    <div style="background:#12121a;border:1px solid #2a2a3a;border-top:2px solid #7c6bff;border-radius:10px;padding:1.2rem"><div style="font-family:Space Mono,monospace;font-size:.62rem;color:#7c6bff;text-transform:uppercase;letter-spacing:.15em;margin-bottom:.4rem">Classification</div><div style="font-size:1.1rem;font-weight:600;margin-bottom:.3rem">ResNet50</div><div style="font-size:.8rem;color:#6b6b80">Feature maps · GradCAM · PCA/t-SNE · Isosurface · Variance · Model comparison</div></div>
    <div style="background:#12121a;border:1px solid #2a2a3a;border-top:2px solid #ff6b9d;border-radius:10px;padding:1.2rem"><div style="font-family:Space Mono,monospace;font-size:.62rem;color:#ff6b9d;text-transform:uppercase;letter-spacing:.15em;margin-bottom:.4rem">Classification</div><div style="font-size:1.1rem;font-weight:600;margin-bottom:.3rem">VGG16</div><div style="font-size:.8rem;color:#6b6b80">5-block visualization · GradCAM · PCA/t-SNE · Isosurface · Variance · Comparison</div></div>
    <div style="background:#12121a;border:1px solid #2a2a3a;border-top:2px solid #6bffd4;border-radius:10px;padding:1.2rem"><div style="font-family:Space Mono,monospace;font-size:.62rem;color:#6bffd4;text-transform:uppercase;letter-spacing:.15em;margin-bottom:.4rem">Detection</div><div style="font-size:1.1rem;font-weight:600;margin-bottom:.3rem">YOLOv5</div><div style="font-size:.8rem;color:#6b6b80">Real-time object detection with EigenCAM activation maps</div></div></div>''', unsafe_allow_html=True)
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
img_np = np.array(image)
h, w = img_np.shape[:2]

col_img, col_info = st.columns([1, 2])
with col_img:
    st.markdown('<div class="section-label">Input Image</div>', unsafe_allow_html=True)
    st.image(image, width=300)
    st.markdown(f'<div style="margin-top:.4rem"><span class="badge">{w}×{h}px</span><span class="badge">RGB</span><span class="badge">{model_name}</span></div>', unsafe_allow_html=True)
with col_info:
    if not run_btn:
        st.markdown('<div style="padding:2rem;color:#6b6b80;font-size:.85rem;line-height:1.8;">← Press <strong style="color:#7c6bff">▶ Run Visualization</strong> in the sidebar to analyze.</div>', unsafe_allow_html=True)

if not run_btn:
    st.stop()

st.markdown("---")

# ══════════════════════════════════════════════════════════
# RESNET50
# ══════════════════════════════════════════════════════════
if "ResNet" in model_choice:
    st.markdown('<div class="section-label">ResNet50 Analysis</div>', unsafe_allow_html=True)
    with st.spinner("Running ResNet50..."):
        results = run_resnet(img_np, show_layers, num_channels)
        resnet_model = get_resnet()
        rgb_img, tensor = resnet_preprocess(img_np)
        top5 = get_top5(resnet_model, tensor, labels)
        with torch.no_grad():
            out = resnet_model(tensor)
        probs = torch.nn.functional.softmax(out[0], dim=0).cpu().numpy()
        pred_metrics = prediction_metrics(probs)
        fmap_eval = resnet_extract(resnet_model, tensor, f"layer{layer_idx}")
        feat_metrics = feature_metrics(fmap_eval)
        cam_metrics = cam_focus_score(results["gradcam"]["grayscale_cam"])

    top_label, top_conf = top5[0]
    stamped = stamp_prediction(results["gradcam"]["overlay"], top_label, top_conf)

    pred_col, info_col = st.columns([1,1])
    with pred_col:
        st.markdown('<div class="section-label">Top-5 Predictions</div>', unsafe_allow_html=True)
        st.markdown(render_predictions(top5), unsafe_allow_html=True)
    with info_col:
        st.markdown('<div class="section-label">Predicted Class</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.2rem;text-align:center"><div style="font-size:2rem;margin-bottom:.3rem">🏷️</div><div style="font-family:Space Mono,monospace;font-size:1rem;color:#7c6bff;margin-bottom:.2rem">{top_label.replace("_"," ").title()}</div><div style="font-size:1.8rem;font-weight:700;color:#e8e8f0">{top_conf*100:.1f}%</div><div style="font-size:.75rem;color:#6b6b80;margin-top:.3rem">confidence</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9= st.tabs([
        "🗺  Feature Maps",
        "🔥  GradCAM",
        "📊  Layer Comparison",
        "📈  Channel Stats",
        "🔵  PCA / t-SNE",
        "〰  Isosurface",
        "📉  Variance",
        "⚡  Model vs Model", 
        "📏  Evaluation",
    ])

    with tab1:
        st.markdown('<div class="info-box">Each cell = one filter. <strong>Bright = high activation</strong>, dark = not responding. Early layers show image-like features; deep layers are abstract.</div>', unsafe_allow_html=True)
        for layer_name, fig in results["feature_maps"].items():
            st.markdown(f'<div class="output-title">{layer_name}</div>', unsafe_allow_html=True)
            cf, cd = st.columns([5,1])
            with cf: st.pyplot(fig)
            with cd: st.download_button("⬇ Save", fig_to_bytes(fig), file_name=f"resnet_{layer_name}.png", mime="image/png", key=f"r_fm_{layer_name}")
            plt.close(fig)

    with tab2:
        st.markdown(f'<div class="info-box"><strong>Red = where ResNet50 looked</strong> to classify as <em>{top_label.replace("_"," ").title()}</em>. If red is on the background, the model may be using the wrong features.</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown('<div class="output-title">Original</div>', unsafe_allow_html=True); st.image(results["gradcam"]["original"], width=280)
        with c2: st.markdown('<div class="output-title">Heatmap</div>', unsafe_allow_html=True); st.image(results["gradcam"]["heatmap"], width=280)
        with c3: st.markdown('<div class="output-title">Overlay + Prediction</div>', unsafe_allow_html=True); st.image(stamped, width=280)
        st.download_button("⬇ Save GradCAM overlay", img_to_bytes(stamped), file_name="resnet_gradcam.png", mime="image/png")

    with tab3:
        st.markdown('<div class="info-box">Spatial resolution <strong>shrinks</strong> (56×56 → 7×7) while channels <strong>grow</strong> (256 → 2048). The network trades spatial detail for semantic abstraction.</div>', unsafe_allow_html=True)
        st.pyplot(results["layer_comparison"])
        st.download_button("⬇ Save", fig_to_bytes(results["layer_comparison"]), file_name="resnet_layer_comparison.png", mime="image/png")
        plt.close(results["layer_comparison"])

    with tab4:
        st.markdown('<div class="info-box"><strong>Mean activation</strong> = how strongly each filter fired. <strong>Sparsity</strong> = % dead/inactive neurons. High sparsity in deep layers is normal — filters become more specialized.</div>', unsafe_allow_html=True)
        fmap = resnet_extract(resnet_model, tensor, f"layer{layer_idx}")
        fmap_np = fmap[0].numpy()
        from resnet_viz import make_feature_map_fig
        sf = make_feature_map_fig(fmap, f"Channel Stats — ResNet50 layer{layer_idx}", num_channels)
        # Use channel stats directly
        plt.style.use('dark_background')
        stats_fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
        stats_fig.patch.set_facecolor('#0a0a0f')
        means = fmap_np.mean(axis=(1,2)); n = min(32, len(means))
        sparsity = (fmap_np < 0.01).mean(axis=(1,2)) * 100
        ax1.bar(range(n), means[:n], color=plt.cm.viridis(means[:n]/(means[:n].max()+1e-8)), width=0.8)
        ax1.set_facecolor('#12121a'); ax1.set_ylabel('Mean Activation', color='#6b6b80', fontsize=8)
        ax1.tick_params(colors='#6b6b80', labelsize=7); ax1.spines[:].set_color('#2a2a3a')
        ax1.set_title(f'Per-Channel Mean — layer{layer_idx}', color='#e8e8f0', fontsize=9, pad=4)
        ax2.bar(range(n), sparsity[:n], color=plt.cm.RdYlGn_r(sparsity[:n]/100), width=0.8)
        ax2.set_facecolor('#12121a'); ax2.set_ylabel('Dead Neurons (%)', color='#6b6b80', fontsize=8)
        ax2.set_xlabel('Channel Index', color='#6b6b80', fontsize=8)
        ax2.tick_params(colors='#6b6b80', labelsize=7); ax2.spines[:].set_color('#2a2a3a')
        ax2.set_title('Channel Sparsity', color='#e8e8f0', fontsize=9, pad=4)
        stats_fig.tight_layout()
        st.pyplot(stats_fig)
        st.download_button("⬇ Save", fig_to_bytes(stats_fig), file_name=f"resnet_stats_layer{layer_idx}.png", mime="image/png")
        plt.close(stats_fig)
        s1, s2, s3 = st.columns(3)
        with s1: st.markdown(stat_card(fmap_np.shape[0], "Total Channels", "#7c6bff"), unsafe_allow_html=True)
        with s2: st.markdown(stat_card(f"{fmap_np.mean():.4f}", "Overall Mean", "#6bffd4"), unsafe_allow_html=True)
        with s3: st.markdown(stat_card(f"{(fmap_np<0.01).mean()*100:.1f}%", "Sparsity", "#ff6b9d"), unsafe_allow_html=True)

    # ── NEW TAB: PCA / t-SNE ─────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-label">Dimensionality Reduction of Channel Activations</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box"><strong>Each dot = one filter/channel</strong>. Channels close together learned similar features. <strong>PCA</strong> is linear and fast — it shows the main directions of variation. <strong>t-SNE</strong> is non-linear and reveals hidden clusters of filters that fire on similar patterns. Color = mean activation strength.</div>', unsafe_allow_html=True)

        pca_layer = st.selectbox("Layer for PCA/t-SNE", ["layer1","layer2","layer3","layer4"], index=layer_idx-1, key="r_pca_layer")
        fast_mode = st.checkbox("Fast mode", value=True, key="r_fast_tsne")
        max_ch = st.slider("Max channels for t-SNE", 64, 512, 256, 32, key="r_max_tsne")

        with st.spinner("Computing PCA + t-SNE (may take longer if Fast mode is off)..."):
            fmap_pca = resnet_extract(resnet_model, tensor, pca_layer)
            pca_fig = make_pca_tsne(
                fmap_pca,
                layer_label=f"ResNet50 · {pca_layer}",
                fast_mode=fast_mode,
                max_channels=max_ch
            )
        st.pyplot(pca_fig)
        st.download_button("⬇ Save PCA/t-SNE", fig_to_bytes(pca_fig), file_name=f"resnet_pca_tsne_{pca_layer}.png", mime="image/png")
        plt.close(pca_fig)

    # ── NEW TAB: Isosurface ───────────────────────────────────────────────
    with tab6:
        st.markdown('<div class="section-label">Isosurface / Threshold Visualization</div>', unsafe_allow_html=True)
        thresholds = iso_threshold_slider("r_iso")
        iso_layer = st.selectbox("Layer", ["layer1","layer2","layer3","layer4"], index=layer_idx-1, key="r_iso_layer")
        with st.spinner("Generating isosurface..."):
            fmap_iso = resnet_extract(resnet_model, tensor, iso_layer)
            iso_fig = make_isosurface(fmap_iso, layer_label=f"ResNet50 · {iso_layer}", thresholds=thresholds)
        st.pyplot(iso_fig)
        st.download_button("⬇ Save Isosurface", fig_to_bytes(iso_fig), file_name=f"resnet_isosurface_{iso_layer}.png", mime="image/png")
        plt.close(iso_fig)

    # ── NEW TAB: Variance ─────────────────────────────────────────────────
    with tab7:
        st.markdown('<div class="section-label">Mean · Variance · Sparsity Maps</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box"><strong>Variance map</strong> shows which spatial regions have the most variable activations across filters — high variance = the network is uncertain or diverse in its response here. Combined with mean and sparsity, this gives a complete statistical picture of each layer.</div>', unsafe_allow_html=True)
        var_layer = st.selectbox("Layer", ["layer1","layer2","layer3","layer4"], index=layer_idx-1, key="r_var_layer")
        with st.spinner("Computing variance maps..."):
            fmap_var = resnet_extract(resnet_model, tensor, var_layer)
            var_fig = make_variance_map(fmap_var, layer_label=f"ResNet50 · {var_layer}")
        st.pyplot(var_fig)
        st.download_button("⬇ Save Variance Maps", fig_to_bytes(var_fig), file_name=f"resnet_variance_{var_layer}.png", mime="image/png")
        plt.close(var_fig)

    with tab8:
        st.markdown('<div class="info-box">Running <strong>both ResNet50 and VGG16</strong> on your image. Do they agree? Do they look at the same regions?</div>', unsafe_allow_html=True)
        with st.spinner("Running VGG16..."):
            vgg_res = run_vgg(img_np, [("Block5",29)], 16)
            vgg_m = get_vgg(); _, vgg_t = vgg_preprocess(img_np)
            vgg_top5 = get_top5(vgg_m, vgg_t, labels)
        vgg_label, vgg_conf = vgg_top5[0]
        vgg_stamped = stamp_prediction(vgg_res["gradcam"]["overlay"], vgg_label, vgg_conf)
        cr, cv = st.columns(2)
        with cr:
            st.markdown('<div class="compare-header">🔵 ResNet50</div>', unsafe_allow_html=True)
            st.image(stamped, width=300); st.markdown(render_predictions(top5[:3]), unsafe_allow_html=True)
        with cv:
            st.markdown('<div class="compare-header">🟣 VGG16</div>', unsafe_allow_html=True)
            st.image(vgg_stamped, width=300); st.markdown(render_predictions(vgg_top5[:3]), unsafe_allow_html=True)
        agree = "✅ Both models agree!" if top_label == vgg_label else f"⚡ Models disagree — ResNet says <em>{top_label.replace('_',' ').title()}</em>, VGG says <em>{vgg_label.replace('_',' ').title()}</em>"
        st.markdown(f'<div class="info-box" style="margin-top:1rem">{agree}</div>', unsafe_allow_html=True)
    
    with tab9:
        st.markdown('<div class="section-label">Quantitative Proxy Metrics</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">These are <strong>proxy metrics</strong>, not a perfect ground-truth interpretability score. They help quantify prediction certainty, attention concentration, and internal feature behavior.</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Top-1 Confidence", f"{pred_metrics['top1_confidence']*100:.2f}%")
            st.metric("Top-1 / Top-2 Margin", f"{pred_metrics['margin']*100:.2f}%")
        with c2:
            st.metric("Normalized Entropy", f"{pred_metrics['normalized_entropy']:.3f}")
            st.metric("CAM Focus Score", f"{cam_metrics['focus_score']:.3f}")
        with c3:
            st.metric("Center Mass", f"{cam_metrics['center_mass']:.3f}")
            st.metric("Activation Sparsity", f"{feat_metrics['sparsity_pct']:.2f}%")

        st.markdown("""
        **How to interpret these**
        - Higher **confidence** and **margin** → stronger prediction separation
        - Lower **entropy** → less uncertainty
        - Higher **focus score** → attention is concentrated, not diffused
        - Higher **sparsity** → more selective feature activation
        """)

# ══════════════════════════════════════════════════════════
# VGG16
# ══════════════════════════════════════════════════════════
elif "VGG" in model_choice:
    st.markdown('<div class="section-label">VGG16 Analysis</div>', unsafe_allow_html=True)
    with st.spinner("Running VGG16..."):
        results = run_vgg(img_np, show_blocks, num_channels)
        vgg_model = get_vgg(); rgb_img, tensor = vgg_preprocess(img_np)
        top5 = get_top5(vgg_model, tensor, labels)
        with torch.no_grad():
            out = vgg_model(tensor)
        probs = torch.nn.functional.softmax(out[0], dim=0).cpu().numpy()
        pred_metrics = prediction_metrics(probs)

        fmap_eval = vgg_extract(vgg_model, tensor, block_map[f"Block{block_idx}"])
        feat_metrics = feature_metrics(fmap_eval)

        cam_metrics = cam_focus_score(results["gradcam"]["grayscale_cam"])
    top_label, top_conf = top5[0]
    stamped = stamp_prediction(results["gradcam"]["overlay"], top_label, top_conf)

    pred_col, info_col = st.columns([1,1])
    with pred_col:
        st.markdown('<div class="section-label">Top-5 Predictions</div>', unsafe_allow_html=True)
        st.markdown(render_predictions(top5), unsafe_allow_html=True)
    with info_col:
        st.markdown('<div class="section-label">Predicted Class</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.2rem;text-align:center"><div style="font-size:2rem;margin-bottom:.3rem">🏷️</div><div style="font-family:Space Mono,monospace;font-size:1rem;color:#ff6b9d;margin-bottom:.2rem">{top_label.replace("_"," ").title()}</div><div style="font-size:1.8rem;font-weight:700;color:#e8e8f0">{top_conf*100:.1f}%</div><div style="font-size:.75rem;color:#6b6b80;margin-top:.3rem">confidence</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🗺  Feature Maps",
        "🔥  GradCAM",
        "📊  Block Comparison",
        "📈  Channel Stats",
        "🔵  PCA / t-SNE",
        "〰  Isosurface",
        "📉  Variance",
        "⚡  Model vs Model",
        "📏  Evaluation",

    ])

    with tab1:
        st.markdown('<div class="info-box">VGG uses <strong>magma colormap</strong> to distinguish from ResNet. Early blocks clearly show image structure; Block5 is almost fully abstract.</div>', unsafe_allow_html=True)
        for block_name, fig in results["feature_maps"].items():
            st.markdown(f'<div class="output-title">{block_name}</div>', unsafe_allow_html=True)
            cf, cd = st.columns([5,1])
            with cf: st.pyplot(fig)
            with cd: st.download_button("⬇ Save", fig_to_bytes(fig), file_name=f"vgg_{block_name}.png", mime="image/png", key=f"v_fm_{block_name}")
            plt.close(fig)

    with tab2:
        st.markdown(f'<div class="info-box"><strong>Red = where VGG16 looked</strong> to classify as <em>{top_label.replace("_"," ").title()}</em>.</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown('<div class="output-title">Original</div>', unsafe_allow_html=True); st.image(results["gradcam"]["original"], width=280)
        with c2: st.markdown('<div class="output-title">Heatmap</div>', unsafe_allow_html=True); st.image(results["gradcam"]["heatmap"], width=280)
        with c3: st.markdown('<div class="output-title">Overlay + Prediction</div>', unsafe_allow_html=True); st.image(stamped, width=280)
        st.download_button("⬇ Save GradCAM overlay", img_to_bytes(stamped), file_name="vgg_gradcam.png", mime="image/png")

    with tab3:
        st.markdown('<div class="info-box">VGG has <strong>no skip connections</strong> — features build sequentially. Notice how the mean activation maps look different from ResNet at the same depth.</div>', unsafe_allow_html=True)
        st.pyplot(results["block_comparison"])
        st.download_button("⬇ Save", fig_to_bytes(results["block_comparison"]), file_name="vgg_block_comparison.png", mime="image/png")
        plt.close(results["block_comparison"])

    with tab4:
        fmap = vgg_extract(vgg_model, tensor, block_map[f"Block{block_idx}"])
        fmap_np = fmap[0].numpy()
        plt.style.use('dark_background')
        stats_fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
        stats_fig.patch.set_facecolor('#0a0a0f')
        means = fmap_np.mean(axis=(1,2)); n = min(32, len(means))
        sparsity = (fmap_np < 0.01).mean(axis=(1,2)) * 100
        ax1.bar(range(n), means[:n], color=plt.cm.magma(means[:n]/(means[:n].max()+1e-8)), width=0.8)
        ax1.set_facecolor('#12121a'); ax1.set_ylabel('Mean Activation', color='#6b6b80', fontsize=8)
        ax1.tick_params(colors='#6b6b80', labelsize=7); ax1.spines[:].set_color('#2a2a3a')
        ax1.set_title(f'Per-Channel Mean — Block{block_idx}', color='#e8e8f0', fontsize=9, pad=4)
        ax2.bar(range(n), sparsity[:n], color=plt.cm.RdYlGn_r(sparsity[:n]/100), width=0.8)
        ax2.set_facecolor('#12121a'); ax2.set_ylabel('Dead Neurons (%)', color='#6b6b80', fontsize=8)
        ax2.set_xlabel('Channel Index', color='#6b6b80', fontsize=8)
        ax2.tick_params(colors='#6b6b80', labelsize=7); ax2.spines[:].set_color('#2a2a3a')
        ax2.set_title('Channel Sparsity', color='#e8e8f0', fontsize=9, pad=4)
        stats_fig.tight_layout()
        st.pyplot(stats_fig)
        st.download_button("⬇ Save", fig_to_bytes(stats_fig), file_name=f"vgg_stats_block{block_idx}.png", mime="image/png")
        plt.close(stats_fig)
        s1, s2, s3 = st.columns(3)
        with s1: st.markdown(stat_card(fmap_np.shape[0], "Total Channels", "#ff6b9d"), unsafe_allow_html=True)
        with s2: st.markdown(stat_card(f"{fmap_np.mean():.4f}", "Overall Mean", "#6bffd4"), unsafe_allow_html=True)
        with s3: st.markdown(stat_card(f"{(fmap_np<0.01).mean()*100:.1f}%", "Sparsity", "#7c6bff"), unsafe_allow_html=True)

    # ── NEW TAB: PCA / t-SNE ─────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-label">Dimensionality Reduction of Channel Activations</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box"><strong>Each dot = one filter/channel</strong>. Channels close together learned similar features. Color = mean activation strength. <strong>t-SNE</strong> reveals clusters — groups of filters that respond to similar patterns in the image.</div>', unsafe_allow_html=True)
        pca_block = st.selectbox("Block for PCA/t-SNE", ["Block1","Block2","Block3","Block4","Block5"], index=block_idx-1, key="v_pca_block")
        fast_mode = st.checkbox("Fast mode", value=True, key="v_fast_tsne")
        max_ch = st.slider("Max channels for t-SNE", 64, 512, 256, 32, key="v_max_tsne")

        with st.spinner("Computing PCA + t-SNE (may take longer if Fast mode is off)..."):
            fmap_pca = vgg_extract(vgg_model, tensor, block_map[pca_block])
            pca_fig = make_pca_tsne(
                fmap_pca,
                layer_label=f"VGG16 · {pca_block}",
                fast_mode=fast_mode,
                max_channels=max_ch
            )
        st.pyplot(pca_fig)
        st.download_button("⬇ Save PCA/t-SNE", fig_to_bytes(pca_fig), file_name=f"vgg_pca_tsne_{pca_block}.png", mime="image/png")
        plt.close(pca_fig)

    # ── NEW TAB: Isosurface ───────────────────────────────────────────────
    with tab6:
        st.markdown('<div class="section-label">Isosurface / Threshold Visualization</div>', unsafe_allow_html=True)
        thresholds = iso_threshold_slider("v_iso")
        iso_block = st.selectbox("Block", ["Block1","Block2","Block3","Block4","Block5"], index=block_idx-1, key="v_iso_block")
        with st.spinner("Generating isosurface..."):
            fmap_iso = vgg_extract(vgg_model, tensor, block_map[iso_block])
            iso_fig = make_isosurface(fmap_iso, layer_label=f"VGG16 · {iso_block}", thresholds=thresholds)
        st.pyplot(iso_fig)
        st.download_button("⬇ Save Isosurface", fig_to_bytes(iso_fig), file_name=f"vgg_isosurface_{iso_block}.png", mime="image/png")
        plt.close(iso_fig)

    # ── NEW TAB: Variance ─────────────────────────────────────────────────
    with tab7:
        st.markdown('<div class="section-label">Mean · Variance · Sparsity Maps</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box"><strong>Variance map</strong> shows spatial regions where filter responses are most diverse. High variance = the network is making complex, varied distinctions in that region. This completes the statistical picture from your proposal: mean + variance + sparsity.</div>', unsafe_allow_html=True)
        var_block = st.selectbox("Block", ["Block1","Block2","Block3","Block4","Block5"], index=block_idx-1, key="v_var_block")
        with st.spinner("Computing variance maps..."):
            fmap_var = vgg_extract(vgg_model, tensor, block_map[var_block])
            var_fig = make_variance_map(fmap_var, layer_label=f"VGG16 · {var_block}")
        st.pyplot(var_fig)
        st.download_button("⬇ Save Variance Maps", fig_to_bytes(var_fig), file_name=f"vgg_variance_{var_block}.png", mime="image/png")
        plt.close(var_fig)

    with tab8:
        st.markdown('<div class="info-box">Running <strong>both VGG16 and ResNet50</strong>. VGG has no skip connections — does that change where it looks?</div>', unsafe_allow_html=True)
        with st.spinner("Running ResNet50..."):
            r_res = run_resnet(img_np, ["layer4"], 16)
            r_m = get_resnet(); _, r_t = resnet_preprocess(img_np)
            r_top5 = get_top5(r_m, r_t, labels)
        r_label, r_conf = r_top5[0]
        r_stamped = stamp_prediction(r_res["gradcam"]["overlay"], r_label, r_conf)
        cv2c, crc = st.columns(2)
        with cv2c:
            st.markdown('<div class="compare-header">🟣 VGG16</div>', unsafe_allow_html=True)
            st.image(stamped, width=300); st.markdown(render_predictions(top5[:3]), unsafe_allow_html=True)
        with crc:
            st.markdown('<div class="compare-header">🔵 ResNet50</div>', unsafe_allow_html=True)
            st.image(r_stamped, width=300); st.markdown(render_predictions(r_top5[:3]), unsafe_allow_html=True)
        agree = "✅ Both models agree!" if top_label == r_label else f"⚡ Models disagree — VGG says <em>{top_label.replace('_',' ').title()}</em>, ResNet says <em>{r_label.replace('_',' ').title()}</em>"
        st.markdown(f'<div class="info-box" style="margin-top:1rem">{agree}</div>', unsafe_allow_html=True)
   
    with tab9:
        st.markdown('<div class="section-label">Quantitative Proxy Metrics</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">These are <strong>proxy metrics</strong>, not a perfect ground-truth interpretability score. They help quantify prediction certainty, attention concentration, and internal feature behavior.</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Top-1 Confidence", f"{pred_metrics['top1_confidence']*100:.2f}%")
            st.metric("Top-1 / Top-2 Margin", f"{pred_metrics['margin']*100:.2f}%")
        with c2:
            st.metric("Normalized Entropy", f"{pred_metrics['normalized_entropy']:.3f}")
            st.metric("CAM Focus Score", f"{cam_metrics['focus_score']:.3f}")
        with c3:
            st.metric("Center Mass", f"{cam_metrics['center_mass']:.3f}")
            st.metric("Activation Sparsity", f"{feat_metrics['sparsity_pct']:.2f}%")

        st.markdown("""
        **How to interpret these**
        - Higher **confidence** and **margin** → stronger prediction separation
        - Lower **entropy** → less uncertainty
        - Higher **focus score** → attention is concentrated, not diffused
        - Higher **sparsity** → more selective feature activation
        """)
# ══════════════════════════════════════════════════════════
# YOLO
# ══════════════════════════════════════════════════════════
else:
    st.markdown('<div class="section-label">YOLOv5 + EigenCAM Analysis</div>', unsafe_allow_html=True)
    with st.spinner("Running YOLOv5..."):
        yolo_model = get_model()
        results_yolo = yolo_model(img_np)
        boxed_img = draw_boxes(img_np, results_yolo)
        cam_img = run_cam(img_np)
        num_detections = len(results_yolo[0].boxes)

    st.markdown(f'<div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1rem;margin-bottom:1rem;display:flex;align-items:center;gap:1rem"><div style="font-size:2rem">🎯</div><div><div style="font-family:Space Mono,monospace;font-size:1.2rem;color:#6bffd4">{num_detections} object(s) detected</div><div style="font-size:.8rem;color:#6b6b80">YOLOv5s — single-pass detection at 640×640</div></div></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="output-title">Original</div>', unsafe_allow_html=True)
        st.image(img_np, width=380)
    with c2:
        st.markdown('<div class="output-title">Detections</div>', unsafe_allow_html=True)
        st.image(boxed_img, width=380)
        st.download_button("⬇ Save detections", img_to_bytes(boxed_img), file_name="yolo_detections.png", mime="image/png")
    with c3:
        st.markdown('<div class="output-title">EigenCAM</div>', unsafe_allow_html=True)
        st.image(cam_img, width=380)
        st.download_button("⬇ Save EigenCAM", img_to_bytes(cam_img), file_name="yolo_eigencam.png", mime="image/png")

    st.markdown('<div class="info-box" style="margin-top:1rem"><strong>EigenCAM</strong> uses the first principal component of feature maps to show which regions triggered detections. <strong>Warm = high activation.</strong> No class label needed — ideal for detection models with multiple objects.</div>', unsafe_allow_html=True)
