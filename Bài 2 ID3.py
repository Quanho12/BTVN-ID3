import pandas as pd
import numpy as np
import pprint

# 1. Khai báo dữ liệu gốc
dataset_credit = {
    'Độ tuổi': [25, 40, 35, 27, 31, 36, 48, 26, 33, 29, 38, 44, 42, 28, 30], 
    'Hôn nhân': ['Độc thân', 'Đã kết hôn', 'Từng ly hôn', 'Đã kết hôn', 'Độc thân', 'Đã kết hôn', 'Độc thân', 'Đã kết hôn', 'Từng ly hôn', 'Độc thân', 'Đã kết hôn', 'Độc thân', 'Đã kết hôn', 'Độc thân', 'Đã kết hôn'],
    'Sở hữu BĐS': ['Ở cùng bố mẹ', 'Nhà sở hữu', 'Nhà thuê', 'Ở cùng bố mẹ', 'Nhà thuê', 'Nhà sở hữu', 'Nhà thuê', 'Nhà thuê', 'Ở cùng bố mẹ', 'Nhà thuê', 'Nhà sở hữu', 'Nhà sở hữu', 'Nhà sở hữu', 'Nhà thuê', 'Ở cùng bố mẹ'],
    'Thu nhập': [7000000, 18000000, 12000000, 9000000, 6000000, 8000000, 7000000, 8000000, 5000000, 10000000, 15000000, 14000000, 10000000, 7000000, 6000000],
    'Rủi ro tín dụng': [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
}

df_credit = pd.DataFrame(dataset_credit)

# 2. Cắt khoảng bằng pd.cut (Chỉ thực hiện 1 lần duy nhất ở đây)
bins_age = [0, 30, 40, 100]
labels_age = ['<=30', '31-40', '>40']
df_credit['Độ tuổi'] = pd.cut(df_credit['Độ tuổi'], bins=bins_age, labels=labels_age)

bins_income = [0, 8000000, 12000000, 1000000000]
labels_income = ['Thấp', 'Trung bình', 'Cao']
df_credit['Thu nhập'] = pd.cut(df_credit['Thu nhập'], bins=bins_income, labels=labels_income)

# 3. Chuyển lại về chữ để chuẩn bị đưa vào hàm ID3
# 3. Chuyển lại về chữ để chuẩn bị đưa vào hàm ID3
df_credit['Độ tuổi'] = df_credit['Độ tuổi'].astype(str)
df_credit['Thu nhập'] = df_credit['Thu nhập'].astype(str)
df_credit['Rủi ro tín dụng'] = df_credit['Rủi ro tín dụng'].astype(str) # Thêm dòng này

# ================= Các hàm cốt lõi =================
def calculate_entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = 0
    for i in range(len(elements)):
        probability = counts[i] / np.sum(counts)
        entropy -= probability * np.log2(probability)
    return entropy

def calculate_information_gain(data, split_attribute_name, target_name="play"):
    total_entropy = calculate_entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    
    weighted_entropy = 0
    for i in range(len(vals)):
        subset = data[data[split_attribute_name] == vals[i]]
        weight = counts[i] / np.sum(counts)
        weighted_entropy += weight * calculate_entropy(subset[target_name])

    information_gain = total_entropy - weighted_entropy
    return information_gain

def id3(data, original_data, features, target_attribute_name="play", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        item_values = [calculate_information_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        
        for value in np.unique(data[best_feature]):
            sub_data = data[data[best_feature] == value].dropna()
            subtree = id3(sub_data, original_data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
            
        return tree  

# ================= Chạy thuật toán =================
# Lấy danh sách các đặc trưng (bỏ cột cuối cùng là 'Rủi ro tín dụng')
features_credit = df_credit.columns[:-1].tolist()

# Gọi hàm ID3
credit_decision_tree = id3(df_credit, df_credit, features_credit, target_attribute_name="Rủi ro tín dụng")

print("Cấu trúc cây quyết định Rủi ro tín dụng:")
pprint.pprint(credit_decision_tree)