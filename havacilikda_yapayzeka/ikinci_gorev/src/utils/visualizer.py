"""
Pozisyon Görselleştirme Araçları

Referans ve tahmin pozisyonlarını 2D/3D grafikler olarak çizer.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from pathlib import Path


def plot_trajectory_2d(
    estimated: List[Dict],
    reference: Optional[List[Dict]] = None,
    output_path: str = "outputs/visualizations/trajectory_2d.png",
):
    """
    X-Z düzleminde (kuş bakışı) yörünge grafiği çizer.

    Args:
        estimated: [{"x", "y", "z", ...}, ...]
        reference: Opsiyonel referans pozisyonlar
        output_path: Kayıt yolu
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    est_x = [p["x"] for p in estimated]
    est_z = [p["z"] for p in estimated]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(est_x, est_z, "b-", linewidth=1.5, label="Tahmin", alpha=0.8)

    if reference:
        ref_x = [p["x"] for p in reference]
        ref_z = [p["z"] for p in reference]
        ax.plot(ref_x, ref_z, "m-", linewidth=1.5, label="Referans", alpha=0.8)

    ax.set_xlabel("X [m]", fontsize=12)
    ax.set_ylabel("Z [m]", fontsize=12)
    ax.set_title("Yörünge — X-Z Düzlemi", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_error_over_time(
    estimated: List[Dict],
    reference: List[Dict],
    output_path: str = "outputs/visualizations/position_error.png",
):
    """
    X, Y, Z eksenlerindeki hata grafiğini kare numarasına göre çizer.
    (Şekil 13'teki formata benzer)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    frames = list(range(len(estimated)))
    err_x = [est["x"] - ref["x"] for est, ref in zip(estimated, reference)]
    err_y = [est["y"] - ref["y"] for est, ref in zip(estimated, reference)]
    err_z = [est["z"] - ref["z"] for est, ref in zip(estimated, reference)]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(frames, err_x, "r-", linewidth=1, label="X hatası", alpha=0.8)
    ax.plot(frames, err_y, "g-", linewidth=1, label="Y hatası", alpha=0.8)
    ax.plot(frames, err_z, "b-", linewidth=1, label="Z hatası", alpha=0.8)

    ax.set_xlabel("Kare Numarası", fontsize=12)
    ax.set_ylabel("Pozisyon Kayması [m]", fontsize=12)
    ax.set_title("Eksen Bazında Pozisyon Hatası", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trajectory_3d(
    estimated: List[Dict],
    reference: Optional[List[Dict]] = None,
    output_path: str = "outputs/visualizations/trajectory_3d.png",
):
    """3D yörünge grafiği."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    est_x = [p["x"] for p in estimated]
    est_y = [p["y"] for p in estimated]
    est_z = [p["z"] for p in estimated]
    ax.plot(est_x, est_y, est_z, "b-", linewidth=1.2, label="Tahmin")

    if reference:
        ref_x = [p["x"] for p in reference]
        ref_y = [p["y"] for p in reference]
        ref_z = [p["z"] for p in reference]
        ax.plot(ref_x, ref_y, ref_z, "m-", linewidth=1.2, label="Referans")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("3D Yörünge", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
