# 液体Gaの構造因子S(Q)ショルダー構造の再現

二種類のGa原子を仮定した分子動力学シミュレーションによる液体ガリウムの構造因子S(Q)ショルダー構造の再現性検証

## ディレクトリ構成

```
lammps_settings_obata/
├── experiments/                    # 実験データと結果
│   ├── 01_single_component/        # 単一成分LJモデル
│   ├── 02_bimodal_preliminary/     # 二成分モデル予備実験
│   └── 03_bimodal_grid_search/     # ★論文の主要結果
├── data/
│   ├── experimental/               # 実験S(Q)データ
│   └── lammps_inputs/              # LAMMPS入力テンプレート
├── thesis/                         # 論文
│   ├── thesis.tex
│   ├── thesis.pdf
│   └── figures/                    # 論文用の図
└── README.md
```

## 主要な結果

### 最適パラメータ（experiments/03_bimodal_grid_search）
- σ₁₂/σ₁₁ = 1.17
- Ga1比率 = 45%
- R-factor = 0.058
- ショルダー部RMSE = 0.026

### 論文図とデータの対応
| 論文の図 | データソース |
|---------|-------------|
| 図1, 2 | thesis/figures/intro_*.png（緒言説明用） |
| 図3 | experiments/01_single_component/ |
| 図4 | experiments/03_bimodal_grid_search/outputs/analysis/rfactor_heatmap.png |
| 図5 | experiments/03_bimodal_grid_search/outputs/analysis/gallery_all_sq.png |
| 図6 | experiments/03_bimodal_grid_search/outputs/analysis/best_fit_overlay.png |
| 図7 | experiments/03_bimodal_grid_search/outputs/analysis/（ショルダー拡大） |

## 旧ディレクトリ（参照用）
- `hirataken20251122-2/`: 元の実験ディレクトリ（整理前）
- `20251024/`: 初期テスト

---

## LAMMPS Setup (AWS)

### boot
```bash
ssh ubuntu@{ipv4 address} -i 20251114.pem
```

### setup
```bash
sudo apt update && \
sudo apt install -y build-essential cmake git libopenmpi-dev openmpi-bin python3 python3-dev && \
cd ~ && \
git clone -b stable https://github.com/lammps/lammps.git && \
cd lammps && \
rm -rf build && \
mkdir build && \
cd build && \
cmake ../cmake \
  -D PKG_GPU=on \
  -D GPU_API=cuda \
  -D GPU_ARCH=sm_90 \
  -D BUILD_MPI=yes \
  -D PKG_MOLECULE=yes \
  -D PKG_MANYBODY=yes \
  -D PKG_MEAM=yes && \
make -j$(nproc) && \
sudo make install && \
lmp -h
```
