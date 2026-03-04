# PyQUBO 実験リポジトリ

量子アニーリング向け QUBO (Quadratic Unconstrained Binary Optimization) 変換の実装実験。  
D-Wave アカウント不要。ローカルの古典シミュレーテッドアニーリング (Neal) で完全動作します。

---

## 構成

```
qubo_experiments/
├── 01_knapsack_qubo.py       # Level 1: ナップサック問題 → QUBO → SA求解
└── 02_surrogate_to_qubo.py   # Level 2: 代理モデル係数 → QUBO → SA求解
```

---

## 環境構築

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac / Linux

pip install pyqubo dwave-neal numpy scikit-learn pandas dimod
```

---

## 実験一覧

### Level 1 — ナップサック問題 (`01_knapsack_qubo.py`)

**何をやっているか**

| ステップ | 内容 |
|---|---|
| 1 | アイテムの価値・重量を定義 |
| 2 | PyQUBO で目的関数 + 重量制約を QUBO 行列に変換 |
| 3 | Neal (古典 SA) で 1000 回サンプリング |
| 4 | 全探索の最適解と比較し Gap を計算 |

```bash
python qubo_experiments/01_knapsack_qubo.py
```

**実行結果例**

```
==================================================
  ナップサック問題 (PyQUBO + Neal SA)
==================================================
容量: 5

選択アイテム:
  item0: 価値=10, 重量=3
  item3: 価値=4, 重量=1
  item4: 価値=3, 重量=1

合計価値  : 17
合計重量  : 5  (容量 5 OK)
制約違反  : {}
エネルギー: -17.0000

--- 全探索による最適解 ---
  item0: 価値=10, 重量=3
  item3: 価値=4, 重量=1
  item4: 価値=3, 重量=1
最適価値: 17
Gap from optimal: 0.00%
```

**確認できること**

- PyQUBO を用いた QUBO 変換の基本フロー
- slack 変数による不等式制約 (`≤`) の等式制約化
- 古典 SA (Neal) による QUBO 最適化

---

### Level 2 — 代理モデル係数 → QUBO 変換 (`02_surrogate_to_qubo.py`)

**何をやっているか**

| ステップ | 内容 |
|---|---|
| 1 | バイナリ変数に対するブラックボックス関数をサンプリング |
| 2 | `PolynomialFeatures` + `LinearRegression` で二次代理モデルを学習 |
| 3 | 回帰係数を QUBO 行列 `{(xi, xj): coeff}` に変換 (`xi² = xi` を適用) |
| 4 | Neal で最小化し、全探索の最適値と比較 |

```bash
python qubo_experiments/02_surrogate_to_qubo.py
```

**実行結果例**

```
=======================================================
  代理モデル係数 → QUBO 変換 (Surrogate-to-QUBO)
=======================================================
変数数     : 4
学習サンプル: 60 点

代理モデルの学習完了 (R² = 1.0000)

QUBO 行列 (非ゼロ要素):
要素                           係数
--------------------------------
  (x0, x0)               3.0000
  (x0, x1)              -6.0000
  (x0, x2)               3.0000
  (x1, x1)               5.0000
  (x1, x3)              -4.0000
  (x2, x2)               2.0000
  (x2, x3)               2.0000
  (x3, x3)               4.0000

SA が選んだ解: x = [0, 0, 0, 0]
  → 真のコスト: 0.0000

全探索の最適解: x = [0, 0, 0, 0]
  → 最適コスト: 0.0000

Gap from optimal: 0.00%
  → 代理モデル経由でも全探索と同じ最適解に到達しました！
```

**確認できること**

- 「代理モデル係数から QUBO 行列を組み立てる」フローの実証
- バイナリ変数の性質 $x_i^2 = x_i$ を利用した高次項の対角要素への繰り込み
- 代理モデル経由でも全探索と同等の最適解が得られること

---

## 技術的背景

### QUBO とは

QUBO は以下の形式の二値最適化問題です：

$$\min_{x \in \{0,1\}^n} \sum_{i} Q_{ii} x_i + \sum_{i < j} Q_{ij} x_i x_j$$

量子アニーリング (D-Wave) や古典 SA はこの形式を直接解くことができます。

### 代理モデル → QUBO の変換原理

二次回帰で得られた関数：

$$f(x) \approx \sum_i a_i x_i + \sum_{i<j} b_{ij} x_i x_j + \text{const}$$

バイナリ変数の性質 $x_i^2 = x_i$ により、`xi^2` の係数は対角要素 `(xi, xi)` に繰り込まれます。  
結果として QUBO 行列は：

$$Q_{ii} = a_i + c_{ii}, \quad Q_{ij} = b_{ij} \quad (i < j)$$

---

## 依存ライブラリ

| ライブラリ | 用途 |
|---|---|
| `pyqubo` | QUBO モデルの記述・コンパイル |
| `dwave-neal` | 古典シミュレーテッドアニーリング |
| `dimod` | BQM (Binary Quadratic Model) の操作 |
| `numpy` | 数値計算 |
| `scikit-learn` | 代理モデル (二次回帰) の学習 |
| `pandas` | 結果の表形式出力 |
