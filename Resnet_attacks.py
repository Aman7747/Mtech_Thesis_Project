import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import time
from scipy import linalg
import pandas as pd
from torchvision.models import inception_v3


class DWT(nn.Module):

    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        x_out = torch.zeros(b, c * 4, h // 2, w // 2, device=x.device)

        for i in range(c):
            # Apply Haar wavelet transform
            ll = (x[:, i, 0::2, 0::2] + x[:, i, 1::2, 0::2] +
                  x[:, i, 0::2, 1::2] + x[:, i, 1::2, 1::2]) / 2
            lh = (x[:, i, 0::2, 0::2] - x[:, i, 1::2, 0::2] +
                  x[:, i, 0::2, 1::2] - x[:, i, 1::2, 1::2]) / 2
            hl = (x[:, i, 0::2, 0::2] + x[:, i, 1::2, 0::2] -
                  x[:, i, 0::2, 1::2] - x[:, i, 1::2, 1::2]) / 2
            hh = (x[:, i, 0::2, 0::2] - x[:, i, 1::2, 0::2] -
                  x[:, i, 0::2, 1::2] + x[:, i, 1::2, 1::2]) / 2

            x_out[:, i, :, :] = ll
            x_out[:, c + i, :, :] = lh
            x_out[:, 2 * c + i, :, :] = hl
            x_out[:, 3 * c + i, :, :] = hh

        return x_out


class IDWT(nn.Module):

    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        # x: (B, 4C, H, W)
        b, c4, h, w = x.shape
        c = c4 // 4
        x_out = torch.zeros(b, c, h * 2, w * 2, device=x.device)

        for i in range(c):
            ll = x[:, i, :, :]
            lh = x[:, c + i, :, :]
            hl = x[:, 2 * c + i, :, :]
            hh = x[:, 3 * c + i, :, :]

            x_out[:, i, 0::2, 0::2] = (ll + lh + hl + hh) / 2
            x_out[:, i, 1::2, 0::2] = (ll - lh + hl - hh) / 2
            x_out[:, i, 0::2, 1::2] = (ll + lh - hl - hh) / 2
            x_out[:, i, 1::2, 1::2] = (ll - lh - hl + hh) / 2

        return x_out


class DenseBlock(nn.Module):

    def __init__(self, in_channels, growth_rate=32, num_layers=3):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate,
                              kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)


class InvBlock(nn.Module):

    def __init__(self, channels, split_channels):
        super(InvBlock, self).__init__()
        self.split_channels = split_channels
        ub_channels = channels - split_channels

        # Affine coupling functions
        self.phi1 = DenseBlock(ub_channels, growth_rate=32, num_layers=2)
        self.phi2 = DenseBlock(ub_channels, growth_rate=32, num_layers=2)
        self.phi3 = DenseBlock(split_channels, growth_rate=32, num_layers=2)
        self.phi4 = DenseBlock(split_channels, growth_rate=32, num_layers=2)

        # Get output channels from dense blocks
        phi1_out = ub_channels + 2 * 32
        phi2_out = ub_channels + 2 * 32
        phi3_out = split_channels + 2 * 32
        phi4_out = split_channels + 2 * 32

        # Projection layers
        self.proj1 = nn.Conv2d(phi1_out, split_channels, 1)
        self.proj2 = nn.Conv2d(phi2_out, split_channels, 1)
        self.proj3 = nn.Conv2d(phi3_out, ub_channels, 1)
        self.proj4 = nn.Conv2d(phi4_out, ub_channels, 1)

        # Learnable scaling factor for stability
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        ua, ub = x[:, :self.split_channels], x[:, self.split_channels:]

        # Stabilized forward coupling
        s1 = self.scale * torch.tanh(self.proj1(self.phi1(ub)))
        exp1 = torch.exp(s1)
        ua_new = ua * exp1 + self.proj2(self.phi2(ub))

        s3 = self.scale * torch.tanh(self.proj3(self.phi3(ua_new)))
        exp3 = torch.exp(s3)
        ub_new = ub * exp3 + self.proj4(self.phi4(ua_new))

        return torch.cat([ua_new, ub_new], dim=1)

    def inverse(self, x):
        ua_new, ub_new = x[:, :self.split_channels], x[:, self.split_channels:]

        # Stabilized inverse coupling
        s3 = self.scale * torch.tanh(self.proj3(self.phi3(ua_new)))
        exp3 = torch.exp(-s3)  # Use exp(-s) for inverse
        ub = (ub_new - self.proj4(self.phi4(ua_new))) * exp3

        s1 = self.scale * torch.tanh(self.proj1(self.phi1(ub)))
        exp1 = torch.exp(-s1)  # Use exp(-s) for inverse
        ua = (ua_new - self.proj2(self.phi2(ub))) * exp1

        return torch.cat([ua, ub], dim=1)


class HAINN(nn.Module):

    def __init__(self, num_blocks=2, channels=3):
        super(HAINN, self).__init__()
        self.channels = channels

        # Wavelet transform layers
        self.dwt = DWT()
        self.idwt = IDWT()

        # Invertible blocks
        self.inv_blocks = nn.ModuleList()
        current_channels = channels * 4

        for i in range(num_blocks):
            split_ch = channels  # First 3 channels for low-freq
            self.inv_blocks.append(InvBlock(current_channels, split_ch))

    def forward(self, x):

        # Apply wavelet transform
        x = self.dwt(x)

        # Pass through invertible blocks
        for block in self.inv_blocks:
            x = block(x)

        # Split into low-freq (first 3 channels) and high-freq (rest)
        x_lr = x[:, :self.channels]  # Low-resolution/Low-frequency
        z = x[:, self.channels:]  # High-frequency latent

        return x_lr, z

    def inverse(self, x_lr, z_adv):

    q
    x = torch.cat([x_lr, z_adv], dim=1)

    # Pass through invertible blocks in reverse
    for block in reversed(self.inv_blocks):
        x = block.inverse(x)

    # Apply inverse wavelet transform
    x_adv = self.idwt(x)

    return x_adv


class FGSM:

    def __init__(self, model, epsilon=16 / 255):
        self.model = model
        self.epsilon = epsilon

    def generate(self, images, labels):
        images.requires_grad = True
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)

        self.model.zero_grad()
        loss.backward()

        perturbation = self.epsilon * images.grad.sign()
        adv_images = images + perturbation
        adv_images = torch.clamp(adv_images, 0, 1)

        return adv_images.detach()


class BIM:

    def __init__(self, model, epsilon=8 / 255, alpha=2 / 255, iterations=30):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations

    def generate(self, images, labels):
        adv_images = images.clone().detach()

        for _ in range(self.iterations):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            loss = F.cross_entropy(outputs, labels)

            self.model.zero_grad()
            loss.backward()

            adv_images = adv_images + self.alpha * adv_images.grad.sign()
            delta = torch.clamp(adv_images - images, -self.epsilon, self.epsilon)
            adv_images = torch.clamp(images + delta, 0, 1).detach()

        return adv_images


class PGD:

    def __init__(self, model, epsilon=8 / 255, alpha=2 / 255, iterations=30):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations

    def generate(self, images, labels):
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
        adv_images = torch.clamp(adv_images, 0, 1).detach()

        for _ in range(self.iterations):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            loss = F.cross_entropy(outputs, labels)

            self.model.zero_grad()
            loss.backward()

            adv_images = adv_images + self.alpha * adv_images.grad.sign()
            delta = torch.clamp(adv_images - images, -self.epsilon, self.epsilon)
            adv_images = torch.clamp(images + delta, 0, 1).detach()

        return adv_images


class DIM:

    def __init__(self, model, epsilon=8 / 255, alpha=2 / 255, iterations=30, diversity_prob=0.5):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.diversity_prob = diversity_prob

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * 1.1)

        if np.random.rand() < self.diversity_prob:
            rnd = np.random.randint(img_size, img_resize)
            rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
            h_rem = img_resize - rnd
            w_rem = img_resize - rnd
            pad_top = np.random.randint(0, h_rem)
            pad_bottom = h_rem - pad_top
            pad_left = np.random.randint(0, w_rem)
            pad_right = w_rem - pad_left

            padded = F.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom], value=0)
            return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)
        else:
            return x

    def generate(self, images, labels):
        adv_images = images.clone().detach()

        for _ in range(self.iterations):
            adv_images.requires_grad = True
            outputs = self.model(self.input_diversity(adv_images))
            loss = F.cross_entropy(outputs, labels)

            self.model.zero_grad()
            loss.backward()

            adv_images = adv_images + self.alpha * adv_images.grad.sign()
            delta = torch.clamp(adv_images - images, -self.epsilon, self.epsilon)
            adv_images = torch.clamp(images + delta, 0, 1).detach()

        return adv_images


class FIDCalculator:

    def __init__(self, device):
        self.device = device
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception_model.eval()
        self.inception_model.fc = nn.Identity()

    def get_activations(self, images):
        with torch.no_grad():
            # Resize to 299x299 for Inception
            if images.shape[-1] != 299:
                images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
            pred = self.inception_model(images)
        return pred.cpu().numpy()

    def calculate_fid(self, real_images, fake_images):
        act1 = self.get_activations(real_images)
        act2 = self.get_activations(fake_images)

        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid


class ImageNetMiniDataset(Dataset):

    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        print(f"Loading dataset from: {root_dir}")

        # Try multiple possible structures
        possible_dirs = [
            os.path.join(root_dir, 'train'),
            os.path.join(root_dir, 'imagenet-mini', 'train'),
            root_dir
        ]

        train_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
                if len(subdirs) > 0:
                    train_dir = dir_path
                    print(f"Found dataset directory: {train_dir}")
                    print(f"Number of classes found: {len(subdirs)}")
                    break

        if train_dir is None:

            print(f"Contents of {root_dir}:")
            for item in os.listdir(root_dir):
                print(f"  - {item}")
            raise ValueError(f"Could not find valid dataset structure in {root_dir}")

        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        print(f"Loading images from {len(classes)} classes...")
        for class_name in tqdm(classes, desc="Loading classes"):
            class_dir = os.path.join(train_dir, class_name)
            if os.path.isdir(class_dir):
                images_in_class = [img for img in os.listdir(class_dir)
                                   if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

                for img_name in images_in_class:
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

        print(f"Total images loaded: {len(self.image_paths)}")

        if len(self.image_paths) == 0:
            raise ValueError("No images found in dataset!")

        if max_samples and max_samples < len(self.image_paths):
            indices = np.random.choice(len(self.image_paths), max_samples, replace=False)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            print(f"Sampled {max_samples} images for training")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


def train_hainn(model, target_model, train_loader, device, epochs=10, lr=5e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # Loss weights
    lambda1, lambda2, lambda3 = 1.0, 1.0, 2.0

    model.train()
    target_model.eval()

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            # Get low-resolution ground truth
            images_lr = F.interpolate(images, scale_factor=0.5,
                                      mode='bicubic', align_corners=False)

            # Forward pass
            x_lr, z = model(images)

            # Sample adversarial latent variables
            z_adv = torch.randn_like(z)

            # Generate adversarial examples
            x_adv = model.inverse(x_lr, z_adv)

            # Compute losses
            loss_frow = F.l1_loss(x_lr, images_lr)

            z_prime = torch.randn_like(z)
            x_recon = model.inverse(x_lr, z_prime)
            loss_back = F.l1_loss(x_recon, images)

            with torch.no_grad():
                logits_adv = target_model(x_adv)
            loss_adv = -F.cross_entropy(logits_adv, labels)

            # Total loss
            loss = lambda1 * loss_frow + lambda2 * loss_back + lambda3 * loss_adv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')

    return model


def evaluate_attack(attack_method, attack_name, model, test_loader, device, fid_calculator, iterations=None):
    """Evaluate a single attack method"""
    print(f"\nEvaluating {attack_name}...")

    total_correct_clean = 0
    total_correct_adv = 0
    total_samples = 0
    total_time = 0

    all_clean_images = []
    all_adv_images = []

    model.eval()

    for images, labels in tqdm(test_loader, desc=attack_name):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)

        # Clean accuracy
        with torch.no_grad():
            outputs_clean = model(images)
            pred_clean = outputs_clean.argmax(dim=1)
            total_correct_clean += (pred_clean == labels).sum().item()

        start_time = time.time()

        if attack_name == "HA-INN":
            adv_images = attack_method(images, labels)
        else:
            adv_images = attack_method.generate(images, labels)

        end_time = time.time()
        total_time += (end_time - start_time)

        with torch.no_grad():
            outputs_adv = model(adv_images)
            pred_adv = outputs_adv.argmax(dim=1)
            total_correct_adv += (pred_adv == labels).sum().item()

        total_samples += batch_size

        all_clean_images.append(images.cpu())
        all_adv_images.append(adv_images.cpu())

        if total_samples >= 500:
            break

    clean_acc = total_correct_clean / total_samples * 100
    adv_acc = total_correct_adv / total_samples * 100
    asr = 100 - adv_acc
    avg_time = total_time / len(all_clean_images)

    all_clean_images = torch.cat(all_clean_images, dim=0).to(device)
    all_adv_images = torch.cat(all_adv_images, dim=0).to(device)
    fid_score = fid_calculator.calculate_fid(all_clean_images, all_adv_images)

    return {
        'Attack Method': attack_name,
        'Iterations': iterations if iterations else 'N/A',
        'RunTime (s)': f'{avg_time:.2f}',
        'ASR (%)': f'{asr:.2f}',
        'FID Score': f'{fid_score:.2f}'

    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    print('Loading dataset...')

    dataset_root = '/kaggle/input/imagenetmini-1000'
    print(f"\nExploring dataset structure at: {dataset_root}")
    print("Directory contents:")
    for root, dirs, files in os.walk(dataset_root):
        level = root.replace(dataset_root, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if level < 2:  # Only show first 2 levels
            subindent = ' ' * 2 * (level + 1)
            for file in files[:3]:  # Show first 3 files
                print(f'{subindent}{file}')
            if len(files) > 3:
                print(f'{subindent}... and {len(files) - 3} more files')
        if level >= 1:
            break

    try:
        dataset = ImageNetMiniDataset(
            root_dir=dataset_root,
            transform=transform,
            max_samples=1000
        )
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nTrying alternative approach...")

        all_images = []
        for root, dirs, files in os.walk(dataset_root):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(root, file))

        print(f"Found {len(all_images)} total images")

        if len(all_images) == 0:
            raise ValueError("No images found in the dataset directory!")

        class SimpleImageDataset(Dataset):
            def __init__(self, image_paths, transform):
                self.image_paths = image_paths[:1000]
                self.transform = transform

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                img_path = self.image_paths[idx]
                try:
                    image = Image.open(img_path).convert('RGB')
                except:
                    image = Image.new('RGB', (224, 224))

                if self.transform:
                    image = self.transform(image)

                label = idx % 100
                return image, label

        dataset = SimpleImageDataset(all_images, transform)
        print(f"Created simple dataset with {len(dataset)} images")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty! Please check the dataset path and structure.")

    print(f"\nDataset successfully loaded with {len(dataset)} images")

    train_loader = DataLoader(dataset, batch_size=min(32, len(dataset)),
                              shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size=min(16, len(dataset)),
                             shuffle=False, num_workers=0, pin_memory=True)

    target_model = models.resnet18(pretrained=True).to(device)
    target_model.eval()

    fid_calculator = FIDCalculator(device)

    ha_inn = HAINN(num_blocks=2, channels=3).to(device)
    ha_inn = train_hainn(ha_inn, target_model, train_loader, device,
                         epochs=10, lr=5e-3)

    torch.save(ha_inn.state_dict(), 'ha_inn_model.pth')

    def hainn_attack(images, labels):
        with torch.no_grad():
            x_lr, z = ha_inn(images)
            z_adv = torch.randn_like(z)
            x_adv = ha_inn.inverse(x_lr, z_adv)
            return torch.clamp(x_adv, 0, 1)

    attack_methods = {
        'FGSM': (FGSM(target_model, epsilon=16 / 255), 1),
        'BIM': (BIM(target_model, epsilon=8 / 255, iterations=30), 30),
        'PGD': (PGD(target_model, epsilon=8 / 255, iterations=30), 30),
        'DIM': (DIM(target_model, epsilon=8 / 255, iterations=30), 30),
        'HA-INN': (hainn_attack, 10)
    }

    results = []
    for attack_name, (attack_obj, iterations) in attack_methods.items():
        result = evaluate_attack(attack_obj, attack_name, target_model,
                                 test_loader, device, fid_calculator, iterations)
        results.append(result)

    print('FINAL RESULTS')

    df = pd.DataFrame(results)
    print('\n', df.to_string(index=False))

    # Save results
    df.to_csv('attack_results.csv', index=False)
    print('\nResults saved to attack_results.csv')

    print(f"Best ASR: {df['Attack Method'].iloc[df['ASR (%)'].apply(lambda x: float(x)).argmax()]}")
    print(f"Lowest FID: {df['Attack Method'].iloc[df['FID Score'].apply(lambda x: float(x)).argmin()]}")
    print(f"Fastest: {df['Attack Method'].iloc[df['RunTime (s)'].apply(lambda x: float(x)).argmin()]}")

    return df


if __name__ == '__main__':
    results_df = main()