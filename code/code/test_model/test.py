def reorganize_data(data):
    result_dict = {}

    for item in data:
        dataset_name = item["dataset_name"]
        result = item["result"]

        # 如果 dataset_name 已经在字典中，添加到已有的列表中
        if dataset_name in result_dict:
            result_dict[dataset_name]["dataset_results"].append(result)
        else:
            # 如果 dataset_name 不在字典中，创建新的列表
            result_dict[dataset_name] = {"dataset_results":[result]}

    return result_dict

# 示例数据
data = [
    {"dataset_name": "dataset1", "result": "result1"},
    {"dataset_name": "dataset2", "result": "result2"},
    {"dataset_name": "dataset1", "result": "result3"},
    {"dataset_name": "dataset3", "result": "result4"},
    {"dataset_name": "dataset2", "result": "result5"},
]

# 重新整理数据
reorganized_data = reorganize_data(data)

# 打印结果
print(reorganized_data)
