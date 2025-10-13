

---

#  From Gradients to Reality: Practical White-Box Adversarial Attacks

## Overview

This project investigates **white-box adversarial attacks** — perturbations applied to deep neural network (DNN) inputs to deceive models while remaining nearly invisible to humans. The goal is to make such attacks more **practical**, **efficient**, and **transferable** across architectures.

The study benchmarks classical and adaptive attacks (FGSM, PGD, C&W, and variants) on the **ImageNet-1000** dataset using **ResNet-18**.

---

## Objectives

1. Analyze vulnerabilities in deep neural networks using white-box adversarial attacks.
2. Implement and evaluate key gradient-based attacks: **FGSM**, **PGD**, and **C&W**.
3. Develop and test **PGD variants** (Normal, Momentum, Adaptive).
4. Evaluate attack effectiveness using metrics:

   * **Attack Success Rate (ASR)**
   * **Runtime Efficiency**
   * **Perceptual Quality (FID)**
5. Propose methods to improve practicality, transferability, and runtime performance.

---

##  Attack Methods Implemented

| **Method** | **Description**                         | **Key Advantage**                  |
| ---------- | --------------------------------------- | ---------------------------------- |
| FGSM       | Single-step gradient-based perturbation | Fast but less stealthy             |
| PGD / BIM  | Iterative optimization attack           | High ASR, moderate runtime         |
| C&W        | Optimization-based L₂ attack            | Low distortion, strong attack      |
| MI-FGSM    | Momentum variant of PGD                 | Improved transferability           |
| APGD       | Adaptive step-size PGD                  | Faster and more stable convergence |

---

##  Experimental Setup

* **Dataset:** [ImageNet-1000 (ILSVRC 2012)](https://www.image-net.org/)
* **Victim Model:** ResNet-18 (Pretrained on ImageNet)
* **Perturbation Bound:** ϵ = 8/255
* **Attack Type:** Non-targeted
* **Framework:** PyTorch
* **Hardware:** NVIDIA RTX 3080 GPU

---

##  Results Summary

| **Attack**              | **Iterations** | **Runtime (s)** | **ASR (%)** | **FID (↓)** |
| ----------------------- | -------------- | --------------- | ----------- | ----------- |
| FGSM                    | 1              | 0.01            | 98.72       | 105.64      |
| BIM                     | 30             | 1.44            | 99.98       | 50.12       |
| C&W                     | 10             | 5.12            | 98.50       | 40.10       |
| Normal PGD              | 10             | 14.14           | 100.00      | 31.70       |
| Momentum PGD            | 10             | 14.38           | 100.00      | 40.40       |
| **Adaptive PGD (APGD)** | 10             | **4.79**        | **100.00**  | **0.06**    |

 **APGD** achieves the best trade-off between efficiency, imperceptibility, and attack strength.

---

##  Future Work

**HA-INN (High-Frequency Adversarial Invertible Neural Network):**
Future extensions will explore invertible neural networks for frequency-aware attacks to improve imperceptibility and transferability.

* Adversarial optimization in **latent frequency space**
* Integration with **adaptive PGD**
* Evaluation using perceptual metrics (**FID**, **LPIPS**, **SSIM**)
* Testing against **adversarially trained defenses**



---


##  Dataset Access

* **ImageNet-1000** (ILSVRC 2012):
  🔗 [https://www.image-net.org/](https://www.image-net.org/)
  *You may also access a subset via Kaggle:*
  🔗 [https://www.kaggle.com/c/imagenet-object-localization-challenge/data](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)

---

##  References

1. I. Goodfellow et al., *Explaining and Harnessing Adversarial Examples*, ICLR 2015
2. A. Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks*, ICLR 2018
3. N. Carlini & D. Wagner, *Towards Evaluating the Robustness of Neural Networks*, IEEE S&P 2017
4. F. Croce & M. Hein, *Reliable Evaluation of Adversarial Robustness*, ICML 2020
5. Y. Dong et al., *Boosting Adversarial Attacks with Momentum*, CVPR 2018

---

##  Project Link

GitHub Repository:
👉 [https://github.com/Aman7747/Mtech_Thesis_Project](https://github.com/Aman7747/Mtech_Thesis_Project)

---


