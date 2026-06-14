import pandas as pd
from pathlib import Path

# 原始 fake news 文件所在文件夹
input_dir = Path(r"F:\PY_projects\03_STCLABS\Choquet\choquet_agent_vote_demo\data\raw_data\shortText\TextClassification\newstitle")

# 转换后的总输出文件
output_path = Path(r"F:\PY_projects\03_STCLABS\Choquet\choquet_agent_vote_demo\data\raw_data\shortText\newstitle.csv")

# TASK_NAME = "implicit_sentiment"
# TASK_DESCRIPTION = (
#     "implicit sentiment analysis: infer the sentiment implied by the described event, "
#     "behavior, or outcome, even when no explicit emotional words are present"
# )
TASK_NAME = "short_text"
TASK_DESCRIPTION = (
    "short text classification: identify the category of a short text according to its topic,"
    "intent, or semantic meaning"
)

# 找到文件夹下所有 csv 文件
csv_files = list(input_dir.glob("*.csv"))

all_dfs = []

for file in csv_files:
    print(f"正在处理：{file}")

    # 原始文件无表头，第1列是 label，第2列是 text
    df = pd.read_csv(
        file,
        header=None,
        names=["label", "text"],
        encoding="utf-8"
    )

    # 增加固定列
    df.insert(0, "task_name", TASK_NAME)
    df.insert(1, "task_description", TASK_DESCRIPTION)

    # 统一列顺序
    df = df[["task_name", "task_description", "label", "text"]]

    all_dfs.append(df)

# 合并所有文件
merged_df = pd.concat(all_dfs, ignore_index=True)

# 保存为一个总文件
merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"全部转换完成，已保存到：{output_path}")