import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import VGG16_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

# ── Load once ──────────────────────────────────
_vgg_model = None

def get_vgg():
    global _vgg_model
    if _vgg_model is None:
        _vgg_model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        _vgg_model.eval()
    return _vgg_model

# ── Preprocess ─────────────────────────────────
def preprocess(img_np):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    pil = Image.fromarray(img_np).convert('RGB')
    rgb_img = np.array(pil.resize((224, 224))) / 255.0
    tensor = transform(pil).unsqueeze(0)
    return rgb_img, tensor

# ── Feature maps ───────────────────────────────
def extract_feature_maps(model, tensor, layer_index):
    activations = {}
    def hook_fn(module, input, output):
        activations['out'] = output.detach()
    hook = model.features[layer_index].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(tensor)
    hook.remove()
    return activations['out']

def make_feature_map_fig(fmap, title, num_channels=16):
    fmap_np = fmap[0].numpy()
    n = min(num_channels, fmap_np.shape[0])
    cols = 4
    rows = max(1, n // cols)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.8))
    fig.patch.set_facecolor('#12121a')
    fig.suptitle(title, fontsize=11, color='#e8e8f0',
                 fontfamily='monospace', y=1.01)

    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < n:
            ch = fmap_np[i]
            ch_norm = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
            ax.imshow(ch_norm, cmap='magma')  # different colormap from ResNet for distinction
            ax.set_title(f'Ch {i}', fontsize=7, color='#6b6b80', pad=2)
        ax.axis('off')
        ax.set_facecolor('#12121a')

    fig.tight_layout()
    return fig

# ── GradCAM ────────────────────────────────────
def make_gradcam(model, tensor, rgb_img):
    target_layer = [model.features[28]]
    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=None)[0]

    heatmap_colored = cv2.applyColorMap(
        (grayscale_cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True)

    return {
        "original": (rgb_img * 255).astype(np.uint8),
        "heatmap": heatmap_colored,
        "overlay": overlay,
        "grayscale_cam": grayscale_cam
    }

# ── Block comparison ───────────────────────────
def make_block_comparison(model, tensor):
    all_blocks = [("Block1", 3), ("Block2", 8), ("Block3", 15), ("Block4", 22), ("Block5", 29)]

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 5, figsize=(22, 7))
    fig.patch.set_facecolor('#0a0a0f')
    fig.suptitle('Feature Evolution Across VGG16 Blocks',
                 fontsize=12, color='#e8e8f0', fontfamily='monospace')

    for col, (label, idx) in enumerate(all_blocks):
        fmap = extract_feature_maps(model, tensor, idx)
        fmap_np = fmap[0].numpy()

        ch0 = fmap_np[0]
        ch0_norm = (ch0 - ch0.min()) / (ch0.max() - ch0.min() + 1e-8)
        axes[0][col].imshow(ch0_norm, cmap='magma')
        axes[0][col].set_title(
            f'{label} (layer {idx})\n{fmap_np.shape[0]}ch · {fmap_np.shape[1]}×{fmap_np.shape[2]}',
            fontsize=8, color='#e8e8f0', pad=4)
        axes[0][col].axis('off')

        mean_act = fmap_np.mean(axis=0)
        mean_norm = (mean_act - mean_act.min()) / (mean_act.max() - mean_act.min() + 1e-8)
        axes[1][col].imshow(mean_norm, cmap='hot')
        axes[1][col].set_title('Mean Activation', fontsize=8, color='#6b6b80', pad=4)
        axes[1][col].axis('off')

    fig.tight_layout()
    return fig

# ── Main entry point ───────────────────────────
def run_vgg(img_np, selected_blocks, num_channels=16):
    model = get_vgg()
    rgb_img, tensor = preprocess(img_np)

    feature_maps = {}
    for block_name, layer_idx in selected_blocks:
        fmap = extract_feature_maps(model, tensor, layer_idx)
        fig = make_feature_map_fig(
            fmap,
            f'Feature Maps — VGG16 {block_name} (first {num_channels} channels)',
            num_channels
        )
        feature_maps[block_name] = fig

    gradcam = make_gradcam(model, tensor, rgb_img)
    block_comparison = make_block_comparison(model, tensor)

    return {
        "feature_maps": feature_maps,
        "gradcam": gradcam,
        "block_comparison": block_comparison
    }
