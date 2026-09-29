import pandas as pd
import numpy as np

# Khởi tạo dữ liệu từ bảng
dataset = {
    'outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
    'temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
    'humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
    'wind': ['weak', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'strong'],
    'play': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(dataset)
def calculate_entropy(target_col):
    # Lấy danh sách các giá trị duy nhất và số lượng của chúng (ví dụ: yes: 9, no: 5)
    elements, counts = np.unique(target_col, return_counts=True)
    
    entropy = 0
    # Tính tổng -p * log2(p) cho từng lớp
    for i in range(len(elements)):
        probability = counts[i] / np.sum(counts)
        entropy -= probability * np.log2(probability)
        
    return entropy
def calculate_information_gain(data, split_attribute_name, target_name="play"):
    # Tính Entropy của tập dữ liệu gốc (H(S))
        total_entropy = calculate_entropy(data[target_name])
    
    # Tính Entropy của tập dữ liệu sau khi chia theo thuộc tính (H(x, S))
        vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    
        weighted_entropy = 0
        for i in range(len(vals)):
        # Tách tập dữ liệu con ứng với từng giá trị của thuộc tính (ví dụ: outlook == 'sunny')
            subset = data[data[split_attribute_name] == vals[i]]
        # Tính trọng số (|S_v| / |S|)
            weight = counts[i] / np.sum(counts)
        # Cộng dồn Entropy có trọng số
            weighted_entropy += weight * calculate_entropy(subset[target_name])

    # Gain = H(S) - H(x, S)
        information_gain = total_entropy - weighted_entropy
        return information_gain

def id3(data, original_data, features, target_attribute_name="play", parent_node_class=None):
    # Điều kiện dừng 1: Tất cả các mẫu đều cùng một lớp -> Trả về lớp đó
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    # Điều kiện dừng 2: Tập dữ liệu rỗng -> Trả về lớp phổ biến nhất của tập cha
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    
    # Điều kiện dừng 3: Không còn thuộc tính nào để chia -> Trả về lớp phổ biến nhất hiện tại
    elif len(features) == 0:
        return parent_node_class
    
    # Tiếp tục phân chia:
    else:
        # Xác định lớp phổ biến nhất của node hiện tại để làm parent_node_class cho các nhánh con
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        # Chọn thuộc tính có Information Gain lớn nhất
        item_values = [calculate_information_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        # Khởi tạo cấu trúc cây cho node này
        tree = {best_feature: {}}
        
        # Loại bỏ thuộc tính đã chọn khỏi danh sách các thuộc tính cần xét tiếp
        features = [i for i in features if i != best_feature]
        
        # Phát triển nhánh cho mỗi giá trị của thuộc tính tốt nhất
        for value in np.unique(data[best_feature]):
            # Tạo tập dữ liệu con (Sv)
            sub_data = data[data[best_feature] == value].dropna()
            
            # Đệ quy gọi lại hàm id3 cho nhánh con
            subtree = id3(sub_data, original_data, features, target_attribute_name, parent_node_class)
            
            # Gắn nhánh con vào cây
            tree[best_feature][value] = subtree
            
        return tree   
features = df.columns[:-1].tolist()

# Xây dựng cây
decision_tree = id3(df, df, features)

# In kết quả dạng dictionary định dạng đẹp
import pprint
print("Cấu trúc cây quyết định ID3:")
pprint.pprint(decision_tree)