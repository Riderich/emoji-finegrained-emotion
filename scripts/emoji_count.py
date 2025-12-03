import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== 路径配置（按你的实际绝对路径）=====
csv_path = r"e:\OneDrive - The Chinese University of Hong Kong\College\Course Content\y3\AIST4010\project\emoji-finegrained-emotion\data\text_pairs\train_pairs_2.csv"
out_png = Path(csv_path).parent / "emoji_counts_pie.png"  # 中文说明：饼图输出到CSV同目录

# ===== 读取与列选择 =====
# 中文说明：使用utf-8-sig以兼容可能存在的BOM；读取后做健壮的列名定位（不区分大小写）
df = pd.read_csv(csv_path, encoding="utf-8-sig")
col_map = {c.lower(): c for c in df.columns}  # 中文说明：统一小写映射定位真实列名
name_col = col_map.get("emoji_name")
alt_col = col_map.get("emoji_alt")
if not name_col and not alt_col:
    raise ValueError("CSV缺少 'emoji_name' 或 'emoji_alt' 列，无法统计emoji。")

# ===== 规范化名称函数 =====
def normalize_name(s: str) -> str:
    # 中文说明：去除首尾空白、去除形如[xxx]的括号、统一为字符串
    s = ("" if pd.isna(s) else str(s)).strip()
    if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
        s = s[1:-1].strip()
    return s

# ===== 生成统计键 =====
# 中文说明：优先emoji_name，缺失时回退emoji_alt；过滤空值
names = []
for _, row in df.iterrows():
    nm = normalize_name(row[name_col]) if name_col else ""
    if not nm and alt_col:
        nm = normalize_name(row[alt_col])
    if nm:
        names.append(nm)

if not names:
    raise ValueError("未能从CSV提取到任何emoji名称，请检查数据内容。")

# ===== 计数与小类合并 =====
counts = pd.Series(names).value_counts().sort_values(ascending=False)  # 中文说明：按出现次数排序
total = counts.sum()
proportions = counts / total
minor_mask = proportions < 0.01  # 中文说明：将占比<1%的类别合并为others

major = counts[~minor_mask]
minor_sum = int(counts[minor_mask].sum())
if minor_sum > 0:
    final_counts = pd.concat([major, pd.Series({"others": minor_sum})])
else:
    final_counts = major

# ===== 打印统计结果 =====
print("总样本数：", int(total))
print("\n各emoji计数（未合并前，Top20）：")
print(counts.head(20))
print("\n绘图用计数（已合并others）：")
print(final_counts)

# 输出完整计数CSV（每个emoji的原始计数与占比）
out_counts_csv = Path(csv_path).parent / "emoji_counts.csv"
counts_df = counts.rename_axis("emoji").reset_index(name="count")
counts_df["proportion"] = counts_df["count"] / total
counts_df.to_csv(out_counts_csv, index=False, encoding="utf-8-sig")
print(f"计数CSV已保存：{out_counts_csv}")

# 依据计数筛选训练用数据：仅保留count>300的emoji对应行
keep_emojis = set(counts[counts > 300].index)  # 中文说明：筛选高频emoji集合

def row_name(row):
    # 中文说明：与统计一致的名称规范化逻辑，优先emoji_name，回退emoji_alt
    nm = normalize_name(row[name_col]) if name_col else ""
    if not nm and alt_col:
        nm = normalize_name(row[alt_col])
    return nm

mask = df.apply(lambda r: row_name(r) in keep_emojis, axis=1)
df_filtered = df.loc[mask]
out_filtered_csv = Path(csv_path).parent / "cleaned_combined_count_gt300.csv"
df_filtered.to_csv(out_filtered_csv, index=False, encoding="utf-8-sig")
print(f"筛选后数据集已保存：{out_filtered_csv}（行数：{len(df_filtered)}，类别数：{len(keep_emojis)}）")

# ===== 绘制饼图 =====
# 中文说明：设置中文字体（Windows上常见字体），避免中文标签乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

labels = list(final_counts.index)
sizes = list(final_counts.values)
legend_labels = [f"{lbl} — {int(val)} samples ({val/total*100:.1f}%)" for lbl, val in zip(labels, sizes)]

plt.figure(figsize=(10, 8))  # 中文说明：画布尺寸
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    autopct=lambda pct: f"{pct:.1f}%",
    startangle=90,
    counterclock=False
)
plt.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), title="Legend")
plt.title("Emoji占比饼图（<1%合并为others）", fontsize=14)
plt.tight_layout()
plt.savefig(out_png, dpi=150)
print(f"\n饼图已保存：{out_png}")
# 如需交互查看可取消下一行注释
# plt.show()
