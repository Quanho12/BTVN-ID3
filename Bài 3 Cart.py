import pandas as pd
import numpy as np
import pprint

# 1. Khởi tạo dữ liệu từ bảng
dataset = {
    'outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
    'temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
    'humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
    'wind': ['weak', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'strong'],
    'play': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(dataset)

# ================= Các hàm cốt lõi của CART =================

def calculate_gini(target_col):
    """
    Tính chỉ số Gini cho một tập dữ liệu: Gini = 1 - sum(p_i^2)
    """
    elements, counts = np.unique(target_col, return_counts=True)
    gini = 1.0
    
    for i in range(len(elements)):
        probability = counts[i] / np.sum(counts)
        gini -= probability ** 2
        
    return gini

def calculate_gini_split(data, split_attribute_name, target_name="play"):
    """
    Tính tổng Gini có trọng số sau khi phân nhánh (Gini Split)
    """
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    
    gini_split = 0.0
    for i in range(len(vals)):
        # Tách tập dữ liệu con
        subset = data[data[split_attribute_name] == vals[i]]
        # Tính trọng số (|S_v| / |S|)
        weight = counts[i] / np.sum(counts)
        # Cộng dồn Gini có trọng số
        gini_split += weight * calculate_gini(subset[target_name])
        
    return gini_split

def build_cart_tree(data, original_data, features, target_attribute_name="play", parent_node_class=None):
    """
    Hàm đệ quy xây dựng cây quyết định theo phương pháp CART
    """
    # Điều kiện dừng 1: Tất cả các mẫu đều cùng một lớp
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    # Điều kiện dừng 2: Tập dữ liệu rỗng
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    
    # Điều kiện dừng 3: Không còn thuộc tính nào để chia
    elif len(features) == 0:
        return parent_node_class
    
    # Tiếp tục phân chia:
    else:
        # Xác định lớp phổ biến nhất của node hiện tại
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        # ĐIỂM KHÁC BIỆT CỐT LÕI SO VỚI ID3: Chọn thuộc tính có Gini Split NHỎ NHẤT
        item_values = [calculate_gini_split(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmin(item_values) # Dùng argmin thay vì argmax
        best_feature = features[best_feature_index]
        
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        
        # Phát triển nhánh
        for value in np.unique(data[best_feature]):
            sub_data = data[data[best_feature] == value].dropna()
            subtree = build_cart_tree(sub_data, original_data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
            
        return tree  

# ================= Chạy thuật toán =================

# Lấy danh sách các thuộc tính (loại bỏ cột mục tiêu 'play')
features = df.columns[:-1].tolist()

# Xây dựng cây bằng CART
cart_decision_tree = build_cart_tree(df, df, features)

print("Cấu trúc cây quyết định phân loại (CART - Gini Index):")
pprint.pprint(cart_decision_tree)