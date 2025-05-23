import pickle
import json
# 文件路径
file_path = 'results.pkl'

# 读取pickle文件
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 输出读取的内容
for key,item in data.items():
    df = item
    row_data = df.iloc[2, :5].values
    row_data[4] -= 1
    # 直接格式化，保留两位小数
    formatted_data = " & ".join([f"{x*100:.2f}" for x in row_data])
    latex_line = f"& {formatted_data} \\\\"
    print(key)
    print(latex_line)
    print()
