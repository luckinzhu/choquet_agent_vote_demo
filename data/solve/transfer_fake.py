import pandas as pd
from pathlib import Path

# ======================
# 1. 路径配置
# ======================

# 原始 fake news 文件所在文件夹
input_dir = Path(r"F:\PY_projects\03_STCLABS\Choquet\choquet_agent_vote_demo\data\raw_data\fakenews\no_images\ReCOVery-master")

# 转换后的总输出文件
output_path = Path(r"F:\PY_projects\03_STCLABS\Choquet\choquet_agent_vote_demo\data\raw_data\fakenews\recovery.csv")

# ======================
# 2. 固定任务信息
# ======================

TASK_NAME = "fake_news"

TASK_DESCRIPTION = (
    "fake news detection: determine whether a news article contains fabricated, "
    "misleading, or unverifiable information presented as factual content"
)

# ======================
# 3. 查找所有 CSV 文件
# ======================

csv_files = list(input_dir.glob("*.csv"))

print("输入文件夹：", input_dir)
print("找到 CSV 文件数量：", len(csv_files))
print("找到的文件：")
for f in csv_files:
    print(" -", f)

if not csv_files:
    raise FileNotFoundError(f"没有在该文件夹中找到 CSV 文件：{input_dir}")

# ======================
# 4. 逐个文件转换
# ======================

all_dfs = []

for file in csv_files:
    print(f"\n正在处理：{file}")

    # 读取原始 CSV
    df = pd.read_csv(file, encoding="utf-8")

    # 检查必要字段是否存在
    # required_columns = ["label", "title", "content"]
    # missing_columns = [col for col in required_columns if col not in df.columns]
    required_columns = ["label", "text"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"{file} 缺少必要字段：{missing_columns}")

    # 只保留 label、title、content
    # new_df = df[["label", "title", "content"]].copy()
    new_df = df[["label", "text"]].copy()

    # 增加固定列
    new_df.insert(0, "task_name", TASK_NAME)
    new_df.insert(1, "task_description", TASK_DESCRIPTION)

    # 调整字段顺序
    # new_df = new_df[["task_name", "task_description", "label", "title", "content"]]

    new_df = new_df[["task_name", "task_description", "label", "text"]]

    all_dfs.append(new_df)

# ======================
# 5. 合并并保存
# ======================

merged_df = pd.concat(all_dfs, ignore_index=True)

merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\n全部转换完成！")
print("输出文件：", output_path)
print("总样本数：", len(merged_df))
print("字段：", list(merged_df.columns))