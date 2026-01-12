#!/usr/bin/env python3
"""
緒言用の図を作成するスクリプト
1. 実験S(Q)にショルダー領域をハイライトした図
2. ハードスフィアモデルとの比較図
"""

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# 日本語フォント設定（利用可能な場合）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

# 実験データの読み込み
exp_data = np.loadtxt('../hirataken20251122-2/data/sq_real_data.csv', delimiter=',')
Q_exp = exp_data[:, 0]
SQ_exp = exp_data[:, 1]

# データの重複を除去（Qでソートして平均化）
Q_unique = []
SQ_unique = []
for q in np.unique(Q_exp):
    mask = Q_exp == q
    Q_unique.append(q)
    SQ_unique.append(np.mean(SQ_exp[mask]))
Q_exp = np.array(Q_unique)
SQ_exp = np.array(SQ_unique)


def percus_yevick_sq(Q, sigma=2.7, eta=0.45):
    """
    Percus-Yevick近似によるハードスフィア液体のS(Q)
    eta: packing fraction (体積分率)
    sigma: 粒子直径
    """
    k = Q * sigma

    # Direct correlation function coefficients
    lambda1 = (1 + 2*eta)**2 / (1 - eta)**4
    lambda2 = -(1 + eta/2)**2 / (1 - eta)**4

    # Fourier transform of c(r) for hard spheres
    def c_fourier(k):
        if np.isscalar(k):
            k = np.array([k])
        result = np.zeros_like(k, dtype=float)
        small = np.abs(k) < 1e-6
        large = ~small

        # For small k, use Taylor expansion
        result[small] = -4*np.pi*sigma**3/3 * (lambda1 + 6*eta*lambda2 + eta/2*lambda1)

        # For larger k
        k_l = k[large]
        term1 = lambda1 * (np.sin(k_l) - k_l*np.cos(k_l)) / k_l**3
        term2 = 6*eta*lambda2 * ((k_l**2 - 2)*np.sin(k_l) + 2*k_l*np.cos(k_l) - 2) / k_l**4
        term3 = eta*lambda1/2 * ((k_l**4 - 12*k_l**2 + 24)*np.sin(k_l) +
                                   (24*k_l - 4*k_l**3)*np.cos(k_l) - 24*k_l) / k_l**6
        result[large] = -24*eta/sigma**3 * (term1 + term2 + term3)
        return result

    # Number density
    rho = 6*eta / (np.pi * sigma**3)

    # S(Q) = 1 / (1 - rho * c(k))
    c_k = c_fourier(k)
    S = 1.0 / (1.0 - rho * c_k)
    return S


# 図1: ショルダー領域をハイライトした図
fig1, ax1 = plt.subplots(figsize=(8, 6))

# 実験データをプロット
ax1.plot(Q_exp, SQ_exp, 'ko-', markersize=4, linewidth=1.5, label='Liquid Ga (Exp.)')

# ショルダー領域をシェーディング
shoulder_min, shoulder_max = 2.8, 3.5
ax1.axvspan(shoulder_min, shoulder_max, alpha=0.3, color='orange',
            label=f'Shoulder region\n({shoulder_min}–{shoulder_max} Å$^{{-1}}$)')

# 矢印とアノテーション
# ショルダー部分のS(Q)値を取得
mask = (Q_exp >= 2.9) & (Q_exp <= 3.2)
if np.any(mask):
    Q_shoulder = Q_exp[mask][len(Q_exp[mask])//2]
    SQ_shoulder = SQ_exp[mask][len(SQ_exp[mask])//2]
else:
    Q_shoulder, SQ_shoulder = 3.0, 1.3

ax1.annotate('Shoulder structure',
             xy=(Q_shoulder, SQ_shoulder),
             xytext=(4.5, 1.8),
             fontsize=12,
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# 第一ピークにもアノテーション
peak_idx = np.argmax(SQ_exp)
Q_peak = Q_exp[peak_idx]
SQ_peak = SQ_exp[peak_idx]
ax1.annotate('First peak',
             xy=(Q_peak, SQ_peak),
             xytext=(1.0, 2.2),
             fontsize=12,
             arrowprops=dict(arrowstyle='->', color='blue', lw=2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

ax1.set_xlabel(r'$Q$ (Å$^{-1}$)', fontsize=14)
ax1.set_ylabel(r'$S(Q)$', fontsize=14)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 2.5)
ax1.legend(loc='upper right', fontsize=11)
ax1.set_title('Structure Factor of Liquid Ga: Shoulder Structure', fontsize=14)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig('figures/intro_shoulder_highlight.png', dpi=150, bbox_inches='tight')
print("Saved: figures/intro_shoulder_highlight.png")


# 図2: ハードスフィアモデルとの比較図
fig2, ax2 = plt.subplots(figsize=(8, 6))

# Qの範囲を設定
Q_hs = np.linspace(0.2, 10, 500)

# ハードスフィアS(Q)を計算（eta=0.45で液体Gaの密度に近い）
SQ_hs = percus_yevick_sq(Q_hs, sigma=2.7, eta=0.45)

# プロット
ax2.plot(Q_exp, SQ_exp, 'ko-', markersize=4, linewidth=1.5, label='Liquid Ga (Exp.)')
ax2.plot(Q_hs, SQ_hs, 'b--', linewidth=2, label='Hard-sphere (Percus-Yevick)')

# ショルダー領域をシェーディング
ax2.axvspan(shoulder_min, shoulder_max, alpha=0.2, color='orange')

# 違いを強調する矢印
ax2.annotate('Asymmetric\n(Ga)',
             xy=(3.1, 1.1),
             xytext=(4.5, 1.5),
             fontsize=11,
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

ax2.annotate('Symmetric\n(Hard-sphere)',
             xy=(3.5, 0.55),
             xytext=(5.5, 0.3),
             fontsize=11,
             arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.8))

ax2.set_xlabel(r'$Q$ (Å$^{-1}$)', fontsize=14)
ax2.set_ylabel(r'$S(Q)$', fontsize=14)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 2.5)
ax2.legend(loc='upper right', fontsize=11)
ax2.set_title('Comparison: Liquid Ga vs Hard-sphere Model', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig('figures/intro_hs_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: figures/intro_hs_comparison.png")

print("\nBoth figures created successfully!")
