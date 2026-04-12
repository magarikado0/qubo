"""
Level 2: 代理モデル (二次回帰) の係数 → QUBO 行列への変換と求解

提案書の核心:
  「代理モデルの回帰係数から QUBO 行列を直接組み立て、量子アニーリングへ投入する」

フロー:
  1. ランダムサンプリングで学習データ生成
  2. PolynomialFeatures + LinearRegression で 2次代理モデルを学習
  3. 学習済み係数 → QUBO 辞書 {(xi, xj): coeff} に変換
  4. Neal (古典 SA) で最小化
  5. 全探索の最小値と比較 → Gap from optimal を出力
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from itertools import product
import neal


def build_black_box_params(n_vars: int, rng: np.random.Generator, interaction_density: float = 0.15):
    """Create a reproducible quadratic cost with linear and pairwise terms."""
    weights = rng.uniform(-2.5, 2.5, size=n_vars)
    interactions = {}
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if rng.random() < interaction_density:
                interactions[(i, j)] = rng.uniform(-3.0, 3.0)
    return weights, interactions

# --------------------------------------------------
# 1. 元の評価関数 (ブラックボックスを想定)
# --------------------------------------------------
def black_box(x: np.ndarray, weights: np.ndarray, interactions: dict[tuple[int, int], float]) -> float:
    """
    バイナリベクトル x に対する目的関数 (最小化したい)。
    実際のユースケースでは外部シミュレータ等から得る値。
    例: 加重マルチカット的なコスト
    """
    n = len(x)
    cost = 0.0
    for i, w in enumerate(weights[:n]):
        cost += w * x[i]
    for (i, j), c in interactions.items():
        if i < n and j < n:
            cost += c * x[i] * x[j]
    return cost

n_vars = 20   # バイナリ変数の数（例: 20）

# --------------------------------------------------
# 2. ランダムサンプリングで学習データ生成
# --------------------------------------------------
rng = np.random.default_rng(42)
weights, interactions = build_black_box_params(n_vars, rng)

n_samples = 400
X_train = np.random.randint(0, 2, size=(n_samples, n_vars)).astype(float)
y_train = np.array([black_box(row, weights, interactions) for row in X_train])

print("=" * 55)
print("  代理モデル係数 → QUBO 変換 (Surrogate-to-QUBO)")
print("=" * 55)
print(f"変数数     : {n_vars}")
print(f"学習サンプル: {n_samples} 点")
print(f"相互作用数 : {len(interactions)}")

# --------------------------------------------------
# 3. 二次回帰で代理モデルを学習
# --------------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)
X_poly = poly.fit_transform(X_train)
feature_names = poly.get_feature_names_out([f"x{i}" for i in range(n_vars)])

reg = LinearRegression().fit(X_poly, y_train)
coef = reg.coef_
intercept = reg.intercept_

print(f"\n代理モデルの学習完了 (R² = {reg.score(X_poly, y_train):.4f})")

# --------------------------------------------------
# 4. 係数 → QUBO 行列に変換
# --------------------------------------------------
# バイナリ変数は x^2 = x なので、2次項 xi^2 の係数は 1次項 (対角) へ繰り込む。
# sklearn の特徴量名の例: '1', 'x0', 'x1', 'x0^2', 'x0 x1', ...

import re

qubo = {}
var_name_re = re.compile(r"([a-zA-Z]\w*)\^?(\d*)")

def parse_feature(name: str):
    """特徴量名 → (var_list) に分解。x0^2 → ['x0','x0'], x0 x1 → ['x0','x1']"""
    if name == "1":
        return []
    vars_out = []
    # スペース区切りで分割
    for token in name.split(" "):
        m = re.fullmatch(r"([a-zA-Z][a-zA-Z0-9]*)\^?(\d*)", token)
        if m:
            base  = m.group(1)
            power = int(m.group(2)) if m.group(2) else 1
            vars_out.extend([base] * power)
    return vars_out

for fname, c in zip(feature_names, coef):
    if abs(c) < 1e-10:
        continue
    vars_in = parse_feature(fname)
    if len(vars_in) == 0:
        # 定数項はスキップ
        continue
    elif len(vars_in) == 1:
        # 1次項: (xi, xi) 対角要素
        v = vars_in[0]
        qubo[(v, v)] = qubo.get((v, v), 0.0) + c
    elif len(vars_in) == 2:
        vi, vj = vars_in
        if vi == vj:
            # xi^2 = xi (バイナリ) → 対角要素に繰り込み
            qubo[(vi, vi)] = qubo.get((vi, vi), 0.0) + c
        else:
            # xi*xj → off-diagonal
            key = (vi, vj) if vi < vj else (vj, vi)
            qubo[key] = qubo.get(key, 0.0) + c

print("\nQUBO 行列 (非ゼロ要素):")
print(f"{'要素':<20} {'係数':>10}")
print("-" * 32)
max_print_terms = 60
for idx, ((i, j), v) in enumerate(sorted(qubo.items())):
    if idx >= max_print_terms:
        print(f"  ... ({len(qubo) - max_print_terms} 要素を省略)")
        break
    print(f"  ({i}, {j}){'':<10} {v:>10.4f}")

# --------------------------------------------------
# 5. Neal で求解
# --------------------------------------------------
# neal は dimod BQM を使う。QUBO dict から直接変換。
import dimod

bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

sampler  = neal.SimulatedAnnealingSampler()
response = sampler.sample(bqm, num_reads=2000, num_sweeps=2000)

best_sample = dict(response.first.sample)
best_x = np.array([best_sample.get(f"x{i}", 0) for i in range(n_vars)])
sa_cost = black_box(best_x, weights, interactions)

print(f"\nSA が選んだ解: x = {best_x.astype(int).tolist()}")
print(f"  → 真のコスト: {sa_cost:.4f}")

# --------------------------------------------------
# 6. 全探索で最適解確認（高次元では計算量が爆発するため条件付き）
# --------------------------------------------------
all_x = None
all_cost = None
if n_vars <= 20:
    all_x = np.array(list(product([0, 1], repeat=n_vars)))
    all_cost = np.array([black_box(row, weights, interactions) for row in all_x])
    opt_idx  = np.argmin(all_cost)
    opt_x    = all_x[opt_idx]
    opt_cost = all_cost[opt_idx]

    print(f"\n全探索の最適解: x = {opt_x.astype(int).tolist()}")
    print(f"  → 最適コスト: {opt_cost:.4f}")

    gap = abs(sa_cost - opt_cost) / (abs(opt_cost) + 1e-12) * 100
    print(f"\nGap from optimal: {gap:.2f}%")
    if gap < 1.0:
        print("  → 代理モデル経由でも全探索と同じ最適解に到達しました！")
    else:
        print("  → 差異あり。サンプル数や正則化を調整してみてください。")
else:
    print("\n全探索はスキップ: 2^n が大きすぎるため (n > 20)")
    random_trials = 50000
    random_x = rng.integers(0, 2, size=(random_trials, n_vars))
    random_cost = np.array([black_box(row, weights, interactions) for row in random_x])
    best_random_idx = int(np.argmin(random_cost))
    best_random_cost = float(random_cost[best_random_idx])
    print(f"ランダム探索({random_trials}点) の最良コスト: {best_random_cost:.4f}")
    print(f"SA 改善量 (random_best - sa_cost): {best_random_cost - sa_cost:.4f}")

# --------------------------------------------------
# 7. 結果サマリを DataFrame で表示
# --------------------------------------------------
print("\n--- 全探索コスト一覧 (上位 10 件) ---")
if all_x is not None and all_cost is not None:
    df = pd.DataFrame(all_x, columns=[f"x{i}" for i in range(n_vars)])
    df["cost"] = all_cost
    print(df.nsmallest(10, "cost").to_string(index=False))
else:
    print("n > 20 のため省略（全探索未実施）")
