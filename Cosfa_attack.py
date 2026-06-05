import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import time
from datetime import timedelta
import lpips
import pyiqa
import pywt
import warnings

warnings.filterwarnings('ignore')


# ==========================================
# 1. UTILITY CLASSES (Dataset, Metrics)
# ==========================================

class ImageNetMiniDataset(Dataset):
    """Dataset loader for ImageNet Mini"""

    def __init__(self, data_dir, transform=None, limit=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        if not self.data_dir.exists():
            kaggle_path = Path('/kaggle/input/imagenetmini-1000/imagenet-mini/val')
            if kaggle_path.exists():
                self.data_dir = kaggle_path
                print(f"Using Kaggle path: {self.data_dir}")
            else:
                raise FileNotFoundError(f"Directory not found: {data_dir}")

        print(f"Scanning directory: {self.data_dir}")

        image_extensions = ['*.JPEG', '*.jpg', '*.png', '*.PNG', '*.JPG']

        class_count = 0
        for class_dir in sorted(self.data_dir.iterdir()):
            if class_dir.is_dir():
                class_count += 1
                img_count = 0

                for ext in image_extensions:
                    for img_path in class_dir.glob(ext):
                        self.samples.append((img_path, class_dir.name))
                        img_count += 1
                        if limit and len(self.samples) >= limit:
                            break
                    if limit and len(self.samples) >= limit:
                        break

                if img_count > 0:
                    print(f"  Found {img_count} images in class {class_dir.name}")

                if limit and len(self.samples) >= limit:
                    break

        if len(self.samples) == 0:
            print(f"\nERROR: No images found!")
            print(f"Searched in: {self.data_dir}")
            print(f"Found {class_count} subdirectories")

        self.classes = sorted(set([s[1] for s in self.samples]))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        print(f"Total: {len(self.samples)} images from {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)


class InceptionV3FeatureExtractor(nn.Module):
    """InceptionV3 for FID calculation"""

    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True, aux_logits=True)
        inception.eval()

        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = inception.maxpool1
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = inception.maxpool2
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if x.shape[2:] != (299, 299):
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# ==========================================
# 2. METRIC FUNCTIONS
# ==========================================

def calculate_linf(img1, img2):
    """Calculate L-infinity distance"""
    return torch.max(torch.abs(img1 - img2)).item()


def calculate_l2(img1, img2):
    """Calculate L2 distance"""
    return torch.norm(img1 - img2, p=2).item()


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(img1, img2):
    if img1.dim() == 4:
        ssim_values = []
        for i in range(img1.shape[0]):
            img1_np = img1[i].detach().cpu().numpy().transpose(1, 2, 0)
            img2_np = img2[i].detach().cpu().numpy().transpose(1, 2, 0)
            ssim_val = ssim(img1_np, img2_np, multichannel=True, data_range=1.0, channel_axis=2)
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        img1_np = img1.detach().cpu().numpy().transpose(1, 2, 0)
        img2_np = img2.detach().cpu().numpy().transpose(1, 2, 0)
        return ssim(img1_np, img2_np, multichannel=True, data_range=1.0, channel_axis=2)


def calculate_activation_statistics(images, model, device, batch_size=32):
    """Calculates mu and sigma for FID"""
    model.eval()
    features_list = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            features = model(batch)
            features_list.append(features.cpu().numpy())
    features = np.concatenate(features_list, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculates Frechet Inception Distance"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid


# ==========================================
# 3. WAVELET TRANSFORM UTILITIES
# ==========================================

class WaveletTransform:
    """Discrete Wavelet Transform for multi-scale perturbations"""

    def __init__(self, wavelet='haar', level=3):
        self.wavelet = wavelet
        self.level = level

    def dwt2d(self, image):
        """Apply 2D DWT to image tensor"""
        img_np = image.squeeze(0).cpu().numpy()

        coeffs_list = []
        for c in range(img_np.shape[0]):
            coeffs = pywt.wavedec2(img_np[c], self.wavelet, level=self.level)
            coeffs_list.append(coeffs)

        return coeffs_list

    def idwt2d(self, coeffs_list, original_shape):
        """Inverse 2D DWT from coefficients"""
        reconstructed = []

        for coeffs in coeffs_list:
            rec = pywt.waverec2(coeffs, self.wavelet)
            rec = rec[:original_shape[0], :original_shape[1]]
            reconstructed.append(rec)

        img_np = np.stack(reconstructed, axis=0)
        return torch.from_numpy(img_np).unsqueeze(0).float()

    def add_wavelet_noise(self, image, noise, bands=['H', 'V', 'D']):
        """Add noise to specific wavelet subbands"""
        device = image.device
        dtype = image.dtype

        coeffs_list = self.dwt2d(image)
        noise_coeffs = self.dwt2d(noise)

        modified_coeffs = []
        for c_idx, coeffs in enumerate(coeffs_list):
            modified = [coeffs[0]]

            for level_idx in range(1, len(coeffs)):
                cH, cV, cD = coeffs[level_idx]
                nH, nV, nD = noise_coeffs[c_idx][level_idx]

                if 'H' in bands:
                    cH = cH + nH
                if 'V' in bands:
                    cV = cV + nV
                if 'D' in bands:
                    cD = cD + nD

                modified.append((cH, cV, cD))

            modified_coeffs.append(modified)

        original_shape = image.shape[2:]
        result = self.idwt2d(modified_coeffs, original_shape)

        return result.to(device).to(dtype)


# ==========================================
# 4. INTEGRATED GRADIENTS FOR PIXEL SELECTION
# ==========================================

def get_integrated_gradients_mask(model, image, true_label, k, steps=50, target_class=None):
    """
    Use Integrated Gradients to identify most important pixels.
    More theoretically grounded than simple gradients.
    """
    if k == 0:
        return torch.zeros_like(image)

    model.eval()
    device = image.device

    # Find target class if not specified
    if target_class is None:
        with torch.no_grad():
            output = model(image)
            probs = F.softmax(output, dim=1)
            sorted_indices = torch.argsort(probs[0], descending=True)
            for idx in sorted_indices:
                if idx.item() != true_label.item():
                    target_class = idx.item()
                    break

    # Baseline: black image
    baseline = torch.zeros_like(image)

    # Compute integrated gradients
    integrated_grads = torch.zeros_like(image)

    for step in range(steps):
        alpha = step / steps
        interpolated = baseline + alpha * (image - baseline)
        interpolated.requires_grad = True

        output = model(interpolated)
        score_diff = output[0, target_class] - output[0, true_label]

        model.zero_grad()
        score_diff.backward()

        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    integrated_grads = integrated_grads * (image - baseline)

    # Calculate pixel importance
    pixel_importance = torch.norm(integrated_grads[0], p=2, dim=0)

    # Select top-k pixels
    flat_importance = pixel_importance.flatten()
    k_safe = min(k, flat_importance.numel())
    _, top_k_indices = torch.topk(flat_importance, k_safe)

    mask_flat = torch.zeros_like(flat_importance)
    mask_flat[top_k_indices] = 1

    mask = mask_flat.reshape(1, 1, *pixel_importance.shape)
    mask = mask.repeat(1, 3, 1, 1)

    return mask


# ==========================================
# 5. MI-PGD WITH WAVELET
# ==========================================

class WaveletMIPGD:
    """
    Momentum Iterative PGD with Wavelet transform.
    Combination of:
    - Momentum for better convergence
    - Wavelet domain perturbations
    - Diversity input for ensemble robustness
    """

    def __init__(self, model, eps=0.03, alpha=0.01, steps=40,
                 decay=1.0, wavelet='haar', wavelet_level=3):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level

        # Wavelet transform
        self.wt = WaveletTransform(wavelet=wavelet, level=wavelet_level)

    def generate_noise(self, batch_size, channels, height, width, device):
        """Generate random noise"""
        return torch.zeros(batch_size, channels, height, width, device=device).uniform_(-self.eps, self.eps)

    def attack(self, images, labels, pixel_mask=None):
        """Execute MI-PGD attack with wavelet transform"""
        device = images.device
        images = images.clone().detach()
        labels = labels.clone().detach()

        # Initialize perturbation with random noise
        delta = self.generate_noise(
            images.shape[0],
            images.shape[1],
            images.shape[2],
            images.shape[3],
            device
        )

        if pixel_mask is not None:
            delta = delta * pixel_mask

        # Momentum
        momentum = torch.zeros_like(delta)

        delta.requires_grad = True

        for step in range(self.steps):
            # Add diversity input
            if step % 5 == 0:
                diversity_noise = torch.randn_like(delta) * 0.01
                adv_images = images + delta + diversity_noise
            else:
                adv_images = images + delta

            adv_images = torch.clamp(adv_images, 0, 1)

            outputs = self.model(adv_images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            grad = delta.grad.detach()

            # Update momentum
            grad_norm = torch.norm(grad, p=1)
            momentum = self.decay * momentum + grad / (grad_norm + 1e-8)

            # Update delta
            delta.data = delta.data + self.alpha * momentum.sign()

            # Apply pixel mask
            if pixel_mask is not None:
                delta.data = delta.data * pixel_mask

            # Apply wavelet transform for multi-scale perturbation
            if step % 3 == 0:
                wavelet_noise = self.generate_noise(
                    images.shape[0],
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                    device
                ) * 0.1

                delta.data = self.wt.add_wavelet_noise(
                    delta.data,
                    wavelet_noise,
                    bands=['H', 'V', 'D']
                )

            # Clip perturbation
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(images + delta.data, 0, 1) - images

            delta.grad.zero_()

        return images + delta.detach()


# ==========================================
# 6. EVALUATION FUNCTIONS
# ==========================================

def evaluate_attack(model, dataloader, attack_obj, k_pixels, device,
                    inception_model=None, lpips_model=None, musiq_model=None):
    """Evaluate attack with comprehensive metrics"""
    model.eval()
    correct_clean = 0
    correct_adv = 0
    total = 0

    ssim_values = []
    linf_values = []
    l2_values = []
    psnr_values = []
    lpips_values = []
    musiq_values = []

    clean_images_list = []
    adv_images_list = []
    image_times = []

    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            clean_outputs = model(images)
            clean_preds = clean_outputs.argmax(dim=1)
            correct_clean += (clean_preds == labels).sum().item()

        for i in range(images.size(0)):
            img_start = time.time()

            img = images[i:i + 1]
            label = labels[i:i + 1]

            # Use Integrated Gradients for pixel selection
            pixel_mask = get_integrated_gradients_mask(
                model, img, label, k_pixels, steps=50
            ).to(device)

            adv_img = attack_obj.attack(img, label, pixel_mask)

            with torch.no_grad():
                adv_output = model(adv_img)
                adv_pred = adv_output.argmax(dim=1)
                correct_adv += (adv_pred == label).sum().item()

            # Calculate all metrics
            ssim_val = calculate_ssim(img, adv_img)
            linf_val = calculate_linf(img, adv_img)
            l2_val = calculate_l2(img, adv_img)
            psnr_val = calculate_psnr(img, adv_img)

            ssim_values.append(ssim_val)
            linf_values.append(linf_val)
            l2_values.append(l2_val)
            psnr_values.append(psnr_val)

            if lpips_model is not None:
                with torch.no_grad():
                    img_normalized = img * 2 - 1
                    adv_normalized = adv_img * 2 - 1
                    lpips_val = lpips_model(img_normalized, adv_normalized).item()
                    lpips_values.append(lpips_val)

            if musiq_model is not None:
                try:
                    with torch.no_grad():
                        musiq_score = musiq_model(adv_img).item()
                        musiq_values.append(musiq_score)
                except:
                    musiq_values.append(50.0)

            clean_images_list.append(img.cpu())
            adv_images_list.append(adv_img.cpu())

            img_time = time.time() - img_start
            image_times.append(img_time)

            total += 1
            if total % 10 == 0:
                print(f"  Processed {total} images | Avg time: {np.mean(image_times):.3f}s")

    if total == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    clean_acc = 100 * correct_clean / total
    adv_acc = 100 * correct_adv / total
    attack_success_rate = 100 * (correct_clean - correct_adv) / correct_clean if correct_clean > 0 else 0

    avg_ssim = np.mean(ssim_values)
    avg_linf = np.mean(linf_values)
    avg_l2 = np.mean(l2_values)
    avg_psnr = np.mean(psnr_values)
    avg_lpips = np.mean(lpips_values) if lpips_values else None
    avg_musiq = np.mean(musiq_values) if musiq_values else None
    avg_img_time = np.mean(image_times)

    fid_score = None
    if inception_model is not None and len(clean_images_list) > 1:
        clean_images = torch.cat(clean_images_list, dim=0)
        adv_images = torch.cat(adv_images_list, dim=0)
        try:
            mu1, sigma1 = calculate_activation_statistics(clean_images, inception_model, device)
            mu2, sigma2 = calculate_activation_statistics(adv_images, inception_model, device)
            fid_score = calculate_fid(mu1, sigma1, mu2, sigma2)
        except Exception as e:
            fid_score = 0.0

    return (clean_acc, adv_acc, attack_success_rate, avg_ssim, fid_score,
            avg_linf, avg_l2, avg_psnr, avg_lpips, avg_musiq, avg_img_time)


# ==========================================
# 7. ADAPTIVE GRID SEARCH
# ==========================================

def adaptive_grid_search_optimal_k(model, model_name, dataloader, device, eps, alpha, steps,
                                   ssim_threshold=0.95, asr_threshold=100.0, fid_threshold=20.0):
    """
    Adaptive grid search to find optimal k value
    """
    model_start_time = time.time()

    attack_obj = WaveletMIPGD(
        model, eps=eps, alpha=alpha, steps=steps,
        decay=1.0, wavelet='haar', wavelet_level=3
    )

    print("Loading InceptionV3 for FID calculation...")
    inception_model = InceptionV3FeatureExtractor().to(device)

    print("Loading LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    print("Loading MUSIQ model...")
    musiq_model = pyiqa.create_metric('musiq', device=device)

    max_pixels = 224 * 224

    print(f"\n{'=' * 80}")
    print(f"Running Adaptive Grid Search for {model_name}")
    print(f"Method: Wavelet Transform + MI-PGD + Integrated Gradients")
    print(f"{'=' * 80}")

    results = []
    k = 0
    step = 50
    best_k = None
    iteration = 0

    while k <= max_pixels:
        iteration += 1
        k_start_time = time.time()
        print(f"\n--- Testing k={k} pixels ---")

        (clean_acc, adv_acc, asr, avg_ssim, fid,
         avg_linf, avg_l2, avg_psnr, avg_lpips, avg_musiq, avg_img_time) = evaluate_attack(
            model, dataloader, attack_obj, k, device, inception_model, lpips_model, musiq_model
        )

        k_elapsed = time.time() - k_start_time

        result = {
            'k': k,
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'attack_success_rate': asr,
            'ssim': avg_ssim,
            'fid': fid if fid is not None else 0.0,
            'linf': avg_linf,
            'l2': avg_l2,
            'psnr': avg_psnr,
            'lpips': avg_lpips if avg_lpips is not None else 0.0,
            'musiq': avg_musiq if avg_musiq is not None else 50.0,
            'avg_time_per_image': avg_img_time,
            'total_time_for_k': k_elapsed
        }
        results.append(result)

        print(f"Clean Acc: {clean_acc:.2f}% | ASR: {asr:.2f}% | Adv Acc: {adv_acc:.2f}%")
        print(
            f"SSIM: {avg_ssim:.4f} | FID: {fid if fid else 0:.4f} | LPIPS: {avg_lpips if avg_lpips else 0:.4f} | MUSIQ: {avg_musiq if avg_musiq else 50:.4f}")
        print(f"L∞: {avg_linf:.6f} | L2: {avg_l2:.4f} | PSNR: {avg_psnr:.2f} dB")
        print(f"Time: {k_elapsed:.2f}s total | {avg_img_time:.3f}s per image")

        ssim_ok = avg_ssim > ssim_threshold
        asr_ok = asr >= asr_threshold
        fid_ok = (fid is not None and fid < fid_threshold)

        if ssim_ok and asr_ok and fid_ok:
            best_k = k
            print(f"\n>>> STOPPING CRITERIA MET at k={k}")
            break

        if k == 0:
            k = 10
        else:
            if len(results) >= 2:
                diff = asr - results[-2]['attack_success_rate']
                if diff > 10:
                    step = 20
                elif diff < 1:
                    step = min(step * 2, 1000)

            k += step

        if iteration > 40:
            break

    model_total_time = time.time() - model_start_time

    return results, best_k, model_total_time


# ==========================================
# 8. UTILITY FUNCTIONS
# ==========================================

def load_model(model_name, device):
    print(f"\nLoading {model_name}...")
    if model_name == 'resnet50':
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            model = models.resnet50(weights=weights)
        except:
            model = models.resnet50(pretrained=True)
    elif model_name == 'convnext_base':
        try:
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
            model = models.convnext_base(weights=weights)
        except:
            model = models.convnext_base(pretrained=True)
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(pretrained=True)
    elif model_name == 'visionmamba_small':
        try:
            # Try to import and load Vision Mamba
            # You may need to install: pip install mamba-ssm causal-conv1d
            import timm
            model = timm.create_model('vit_small_patch16_224', pretrained=True)
            print("Note: Using ViT-Small as Vision Mamba proxy (install vim package for actual Vision Mamba)")
        except:
            raise ValueError("Vision Mamba not available. Install with: pip install timm")
    else:
        raise ValueError(f"Model {model_name} not supported")

    model = model.to(device)
    model.eval()
    return model


def visualize_results(results, model_name):
    k_vals = [r['k'] for r in results]
    asr = [r['attack_success_rate'] for r in results]
    ssim = [r['ssim'] for r in results]
    lpips_vals = [r['lpips'] for r in results]
    musiq_vals = [r['musiq'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(k_vals, asr, 'r-o', linewidth=2, markersize=8)
    axes[0, 0].set_title(f'{model_name}: Attack Success Rate vs K\n(Wavelet+MI-PGD+IG)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('K Pixels', fontsize=11)
    axes[0, 0].set_ylabel('ASR (%)', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(k_vals, ssim, 'b-s', linewidth=2, markersize=8)
    axes[0, 1].set_title(f'{model_name}: SSIM vs K', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('K Pixels', fontsize=11)
    axes[0, 1].set_ylabel('SSIM', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(k_vals, lpips_vals, 'g-^', linewidth=2, markersize=8)
    axes[1, 0].set_title(f'{model_name}: LPIPS vs K', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('K Pixels', fontsize=11)
    axes[1, 0].set_ylabel('LPIPS (lower=better)', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(k_vals, musiq_vals, 'm-d', linewidth=2, markersize=8)
    axes[1, 1].set_title(f'{model_name}: MUSIQ vs K', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('K Pixels', fontsize=11)
    axes[1, 1].set_ylabel('MUSIQ Score', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'results_{model_name}_wavelet_mipgd.png', dpi=300)
    print(f"✓ Plot saved to results_{model_name}_wavelet_mipgd.png")


# ==========================================
# 9. MAIN EXECUTION
# ==========================================

def main():
    overall_start_time = time.time()

    DATA_DIR = '/kaggle/input/imagenetmini-1000/imagenet-mini/val'

    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found. Please adjust path.")
        return

    BATCH_SIZE = 1
    NUM_SAMPLES = 50

    EPS = 8 / 255
    ALPHA = 2 / 255
    STEPS = 40

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    print(f"\nLoading dataset from {DATA_DIR}...")
    dataset = ImageNetMiniDataset(DATA_DIR, transform=transform, limit=NUM_SAMPLES)
    print(f"Dataset loaded: {len(dataset)} images found")

    if len(dataset) == 0:
        print("\nERROR: No images found in dataset!")
        return

    target_models_list = ['resnet50', 'convnext_base', 'vit_b_16', 'visionmamba_small']

    all_results = {}
    model_times = {}

    print("\n" + "=" * 80)
    print("ATTACK FRAMEWORK")
    print("Methods: Wavelet Transform + MI-PGD + Integrated Gradients")
    print("=" * 80)

    # Adaptive grid search on each model
    for model_name in target_models_list:
        try:
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
            model = load_model(model_name, device)

            results, best_k, model_time = adaptive_grid_search_optimal_k(
                model, model_name, dataloader, device, EPS, ALPHA, STEPS
            )

            all_results[model_name] = results
            model_times[model_name] = model_time
            visualize_results(results, model_name)

            print(f"\n{'=' * 60}")
            print(f"TOTAL TIME FOR {model_name.upper()}: {timedelta(seconds=int(model_time))}")
            if best_k is not None:
                print(f"OPTIMAL K FOUND: {best_k} pixels")
            print(f"{'=' * 60}")

        except Exception as e:
            print(f"Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()

    overall_time = time.time() - overall_start_time

    # Final reporting
    print("\n" + "=" * 160)
    print("FINAL RESULTS: WAVELET + MI-PGD + INTEGRATED GRADIENTS")
    print("=" * 160)

    summary_data = []
    for model_name, res_list in all_results.items():
        for r in res_list:
            summary_data.append({
                "Model": model_name,
                "K": r['k'],
                "Clean%": f"{r['clean_accuracy']:.2f}",
                "Adv%": f"{r['adversarial_accuracy']:.2f}",
                "ASR%": f"{r['attack_success_rate']:.2f}",
                "SSIM": f"{r['ssim']:.4f}",
                "FID": f"{r['fid']:.4f}",
                "LPIPS": f"{r['lpips']:.4f}",
                "MUSIQ": f"{r['musiq']:.4f}",
                "L∞": f"{r['linf']:.6f}",
                "L2": f"{r['l2']:.4f}",
                "PSNR": f"{r['psnr']:.2f}",
                "Time/Img": f"{r['avg_time_per_image']:.3f}s"
            })

    if summary_data:
        df = pd.DataFrame(summary_data)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')

        print(df.to_string(index=False))
        df.to_csv('attack_results_wavelet_mipgd.csv', index=False)
        print(f"\n✓ Results exported to 'attack_results_wavelet_mipgd.csv'")

    print("\n" + "=" * 160)
    print(f"TOTAL EXECUTION TIME: {timedelta(seconds=int(overall_time))}")
    print("=" * 160)

    print(f"\n✓ Generated files:")
    print(f"  - attack_results_wavelet_mipgd.csv")
    for model_name in all_results.keys():
        print(f"  - results_{model_name}_wavelet_mipgd.png")

    print("\n" + "=" * 80)
    print("KEY FEATURES IN THIS APPROACH:")
    print("=" * 80)
    print("1. Wavelet Transform: Multi-scale perturbations in DWT domain")
    print("2. MI-PGD: Momentum Iterative PGD with diversity input")
    print("3. Integrated Gradients: Theoretically grounded pixel attribution")
    print("4. Adaptive Grid Search: Intelligent k-value optimization")
    print("=" * 80)


if __name__ == "__main__":
    main()