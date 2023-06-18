import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


def result_predict(predict):
    result = ''
    if predict==0:
        result = 'A00'
    elif predict==1:
        result = 'A01'
    elif predict==2:
        result = 'B00'
    elif predict==3:
        result = 'B08'
    elif predict==4:
        result = 'C00'
    else:
        result = 'D00'
    return result

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv('./diem_thpt.csv')

# Chọn các cột đặc trưng
features = ['toan1', 'ly1', 'hoa1', 'sinh1', 'van1', 'su1', 'dia1', 'anh1','hocluc1']

X = data[features]
y = data['hocluc2']

# Chuyển đổi nhãn sang dạng số
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Xây dựng mô hình MLP
mlp = MLPClassifier(hidden_layer_sizes=500)

# Huấn luyện mô hình
mlp.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification = classification_report(y_test, y_pred)

print("Độ chính xác: ", accuracy*100)
print("Báo cáo phân loại:")
print(classification)

# Dự đoán
def predict_grade(scores):
    prediction = mlp.predict(scores)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

# Nhập dữ liệu và dự đoán
def get_valid_score(message):
    while True:
        try:
            score = float(input(message))
            if score < 0 or score > 10:
                print("Điểm phải nằm trong khoảng từ 0 đến 10. Vui lòng nhập lại.")
            else:
                return score
        except ValueError:
            print("Vui lòng nhập một số hợp lệ.")

toan1 = get_valid_score("Nhập điểm toán lần 1: ")
ly1 = get_valid_score("Nhập điểm lý lần 1: ")
hoa1 = get_valid_score("Nhập điểm hoá lần 1: ")
sinh1 = get_valid_score("Nhập điểm sinh lần 1: ")
van1 = get_valid_score("Nhập điểm văn lần 1: ")
su1 = get_valid_score("Nhập điểm sử lần 1: ")
dia1 = get_valid_score("Nhập điểm địa lần 1: ")
anh1 = get_valid_score("Nhập điểm anh lần 1: ")
hocluc1 = float(input("Nhap hoc luc lop 11"))

# Tạo dữ liệu đầu vào
input_scores = [[toan1, ly1, hoa1, sinh1, van1, su1, dia1, anh1, hocluc1]]


# Dự đoán kết quả
predicted_grade = predict_grade(input_scores)
print("Kết quả dự đoán: ", predicted_grade)


# core/






