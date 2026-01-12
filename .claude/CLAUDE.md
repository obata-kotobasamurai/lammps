# Project Instructions

## 役割

あなたの仕事は、フォルダ内の実験データからユーザーの論文の推敲を手伝い、発表資料を作成することです。

## プロジェクト概要

- **テーマ**: 二種類のGa原子を仮定した分子動力学シミュレーションによる液体ガリウムの構造因子S(Q)ショルダー構造の再現性検証
- **論文**: `thesis/thesis.tex`
- **主要な実験結果**: `experiments/03_bimodal_grid_search/`

## 重要なファイル

| ファイル | 説明 |
|---------|------|
| `experiments/README.md` | 実験一覧と概要 |
| `experiments/03_bimodal_grid_search/outputs/analysis/metrics_summary.csv` | パラメータ探索結果 |
| `data/experimental/sq_real_data.csv` | 実験S(Q)データ |
| `thesis/thesis.tex` | 論文本体 |
| `thesis/figures/` | 論文用の図 |

## 最適パラメータ（参照用）

- σ₁₂/σ₁₁ = 1.17
- Ga1比率 = 45%
- R-factor = 0.058
- ショルダー部RMSE = 0.026
