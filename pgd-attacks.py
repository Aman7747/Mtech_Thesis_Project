import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import time
import numpy as np
import os
from tqdm import tqdm
import torchattacks
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

DATA_DIR = '/kaggle/input/imagenetmini-1000'
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ITERATIONS = 10
SUBSET_SIZE = 500

# 1. Data Loading and Preprocessing ---
# ImageNet-mini statistics
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

if SUBSET_SIZE:

    train_size = len(dataset) - SUBSET_SIZE
    val_size = SUBSET_SIZE
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(42))
else:

    val_dataset = dataset

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 2. Model Preparation ---
# Load pretrained ResNet-18
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(DEVICE)
resnet18.eval()


class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(DEVICE)
        self.std = torch.tensor(std).view(1, 3, 1, 1).to(DEVICE)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


model = NormalizedModel(resnet18, mean, std)
model.eval()

# --- 3. FID Score Calculation Setup ---
# FID uses a different model (InceptionV3)
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
fid_model = InceptionV3([block_idx]).to(DEVICE)
fid_model.eval()


def get_activations(loader, model, device):
    pred_arr = []

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Calculating activations"):
            images = images.to(device)
            pred = model(images)[0]

            if isinstance(pred, tuple):
                pred = pred[0]

            pred_arr.append(pred.cpu().numpy().reshape(images.size(0), -1))

    return np.concatenate(pred_arr, axis=0)


# 4. Attack Definitions ---
attacks = {
    "Normal PGD": torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=NUM_ITERATIONS, random_start=True),
    "Momentum PGD (MIFGSM)": torchattacks.MIFGSM(model, eps=8 / 255, alpha=2 / 255, steps=NUM_ITERATIONS, decay=1.0),
    "Adaptive PGD (APGD)": torchattacks.APGD(model, norm='Linf', eps=8 / 255, steps=NUM_ITERATIONS, n_restarts=1,
                                             seed=0, verbose=False)
}

# 5. Evaluation Loop ---
results = {}

# Pre-calculate original activations for FID
original_activations = get_activations(val_loader, fid_model, DEVICE)
mu_orig, sigma_orig = np.mean(original_activations, axis=0), np.cov(original_activations, rowvar=False)

for attack_name, attack in attacks.items():
    print(f"\n--- Evaluating: {attack_name} ---")

    start_time = time.time()
    correct_clean = 0
    successful_attacks_count = 0
    total_correctly_classified = 0

    adv_images_list = []

    for images, labels in tqdm(val_loader, desc=f"Attacking with {attack_name}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Generate adversarial images
        adv_images = attack(images, labels)
        adv_images_list.append(adv_images.cpu())

        # Model predictions
        with torch.no_grad():
            outputs_clean = model(images)
            _, predicted_clean = torch.max(outputs_clean.data, 1)

            outputs_adv = model(adv_images)
            _, predicted_adv = torch.max(outputs_adv.data, 1)

        # We only care about images that were correctly classified initially
        correctly_classified_mask = (predicted_clean == labels)
        total_correctly_classified += correctly_classified_mask.sum().item()

        # An attack is successful if a correctly classified image is now misclassified
        successful_attacks_count += (
                    predicted_adv[correctly_classified_mask] != labels[correctly_classified_mask]).sum().item()

    end_time = time.time()

    # --- Calculate Metrics ---
    runtime = end_time - start_time

    # Attack Success Rate (ASR)
    asr = (successful_attacks_count / total_correctly_classified) * 100 if total_correctly_classified > 0 else 0

    # FID Score
    adv_dataset = torch.utils.data.TensorDataset(torch.cat(adv_images_list, dim=0))
    adv_loader_for_fid = DataLoader(adv_dataset, batch_size=BATCH_SIZE, shuffle=False)


    class FIDLoaderWrapper:
        def __init__(self, loader):
            self.loader_iter = iter(loader)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.loader_iter)[0], None  # Return image, dummy label


    adv_activations = get_activations(FIDLoaderWrapper(adv_loader_for_fid), fid_model, DEVICE)
    mu_adv, sigma_adv = np.mean(adv_activations, axis=0), np.cov(adv_activations, rowvar=False)

    fid_score = calculate_frechet_distance(mu_orig, sigma_orig, mu_adv, sigma_adv)

    results[attack_name] = {
        "Runtime (s)": runtime,
        "Iterations": NUM_ITERATIONS,
        "Attack Success Rate (%)": asr,
        "FID Score": fid_score
    }

    print(f"Results for {attack_name}:")
    print(f"  Runtime: {runtime:.2f} seconds")
    print(f"  Iterations: {NUM_ITERATIONS}")
    print(f"  Attack Success Rate: {asr:.2f}%")
    print(f"  FID Score: {fid_score:.2f}")

print("{:<25} {:<15} {:<12} {:<25}  {:<12}".format("Attack Method", "Runtime (s)", "Iterations",
                                                   "Attack Success Rate (%)", "FID Score"))

for attack_name, metrics in results.items():
    print("{:<25}  {:<15.2f}  {:<12}  {:<25.2f}  {:<12.2f}".format(
        attack_name,
        metrics["Runtime (s)"],
        metrics["Iterations"],
        metrics["Attack Success Rate (%)"],
        metrics["FID Score"]
    ))