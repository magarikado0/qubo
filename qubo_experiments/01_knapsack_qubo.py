"""
Level 1: ナップサック問題を PyQUBO + Neal (古典SA) で解く
D-Wave アカウント不要。ローカルで完全動作します。
"""

from pyqubo import Binary, Constraint
import neal
import numpy as np

# --------------------------------------------------
# 問題設定
# --------------------------------------------------
# アイテム: (名前, 価値, 重量)
items = [
    ("item0", 10, 3),
    ("item1", 6,  2),
    ("item2", 5,  2),
    ("item3", 4,  1),
    ("item4", 3,  1),
]
capacity = 5          # ナップサックの容量
penalty   = 20.0      # 制約違反ペナルティ係数

# --------------------------------------------------
# QUBO モデル構築 (PyQUBO)
# --------------------------------------------------
x = [Binary(f"x{i}") for i in range(len(items))]

# 目的関数: 価値の最大化 → 最小化に符号反転
objective = -sum(v * x[i] for i, (_, v, _) in enumerate(items))

# 制約: 重量 <= capacity
#   slack 変数で等式制約に変換: sum(w*x) + slack = capacity
#   slack を 2進展開
max_slack_bits = int(np.ceil(np.log2(capacity + 1)))
slack_vars = [Binary(f"s{k}") for k in range(max_slack_bits)]
slack = sum((2 ** k) * slack_vars[k] for k in range(max_slack_bits))

weight_sum = sum(w * x[i] for i, (_, _, w) in enumerate(items))
constraint = Constraint(
    (weight_sum + slack - capacity) ** 2,
    label="weight_constraint"
)

model = objective + penalty * constraint
bqm   = model.compile().to_bqm()

# --------------------------------------------------
# Neal (古典シミュレーテッドアニーリング) で求解
# --------------------------------------------------
sampler  = neal.SimulatedAnnealingSampler()
response = sampler.sample(bqm, num_reads=1000, num_sweeps=1000)

# --------------------------------------------------
# 結果デコード
# --------------------------------------------------
best = response.first
decoded_sample = model.compile().decode_sample(
    best.sample, vartype="BINARY"
)
# PyQUBO >= 1.4 では DecodedSample オブジェクトを返す
decoded = decoded_sample.sample
broken  = decoded_sample.constraints(only_broken=True)
energy  = decoded_sample.energy

selected_items = [
    items[i] for i in range(len(items)) if decoded.get("x{}".format(i), 0) == 1
]
total_value  = sum(v for _, v, _ in selected_items)
total_weight = sum(w for _, _, w in selected_items)

print("=" * 50)
print("  ナップサック問題 (PyQUBO + Neal SA)")
print("=" * 50)
print(f"容量: {capacity}")
print()
print("選択アイテム:")
for name, v, w in selected_items:
    print(f"  {name}: 価値={v}, 重量={w}")
print()
print(f"合計価値  : {total_value}")
print(f"合計重量  : {total_weight}  (容量 {capacity} {'OK' if total_weight <= capacity else 'NG - 超過!'})")
print(f"制約違反  : {broken}")
print(f"エネルギー: {energy:.4f}")
print()

# 全探索で最適値を確認
best_val, best_sel = 0, []
for mask in range(1 << len(items)):
    sel = [items[i] for i in range(len(items)) if mask & (1 << i)]
    tw  = sum(w for _, _, w in sel)
    tv  = sum(v for _, v, _ in sel)
    if tw <= capacity and tv > best_val:
        best_val, best_sel = tv, sel

print("--- 全探索による最適解 ---")
for name, v, w in best_sel:
    print(f"  {name}: 価値={v}, 重量={w}")
print(f"最適価値: {best_val}")
gap = (best_val - total_value) / best_val * 100 if best_val else 0
print(f"Gap from optimal: {gap:.2f}%")
