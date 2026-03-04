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

# --------------------------------------------------
# 1. 元の評価関数 (ブラックボックスを想定)
# --------------------------------------------------
def black_box(x: np.ndarray) -> float:
    """
    バイナリベクトル x に対する目的関数 (最小化したい)。
    実際のユースケースでは外部シミュレータ等から得る値。
    例: 加重マルチカット的なコスト
    """
    n = len(x)
    cost = 0.0
    weights = [3.0, 5.0, 2.0, 4.0]   # 1次係数
    interactions = {                    # 2次相互作用
        (0, 1): -6.0,
        (0, 2):  3.0,
        (1, 3): -4.0,
        (2, 3):  2.0,
    }
    for i, w in enumerate(weights[:n]):
        cost += w * x[i]
    for (i, j), c in interactions.items():
        if i < n and j < n:
            cost += c * x[i] * x[j]
    return cost

n_vars = 4   # バイナリ変数の数

# --------------------------------------------------
# 2. ランダムサンプリングで学習データ生成
# --------------------------------------------------
np.random.seed(42)
n_samples = 60
X_train = np.random.randint(0, 2, size=(n_samples, n_vars)).astype(float)
y_train = np.array([black_box(row) for row in X_train])

print("=" * 55)
print("  代理モデル係数 → QUBO 変換 (Surrogate-to-QUBO)")
print("=" * 55)
print(f"変数数     : {n_vars}")
print(f"学習サンプル: {n_samples} 点")

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
for (i, j), v in sorted(qubo.items()):
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
sa_cost = black_box(best_x)

print(f"\nSA が選んだ解: x = {best_x.astype(int).tolist()}")
print(f"  → 真のコスト: {sa_cost:.4f}")

# --------------------------------------------------
# 6. 全探索で最適解確認
# --------------------------------------------------
all_x    = np.array(list(product([0, 1], repeat=n_vars)))
all_cost = np.array([black_box(row) for row in all_x])
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
    print(f"  → 差異あり。サンプル数や正則化を調整してみてください。")

# --------------------------------------------------
# 7. 結果サマリを DataFrame で表示
# --------------------------------------------------
print("\n--- 全探索コスト一覧 (上位 10 件) ---")
df = pd.DataFrame(all_x, columns=[f"x{i}" for i in range(n_vars)])
df["cost"] = all_cost
print(df.nsmallest(10, "cost").to_string(index=False))
