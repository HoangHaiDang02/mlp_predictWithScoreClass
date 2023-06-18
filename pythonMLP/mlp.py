import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold
import tkinter as tk
from tkinter import messagebox




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

data = pd.read_csv('test_2.csv')

# Chọn các cột đặc trưng
features = ['toan1', 'ly1', 'hoa1', 'sinh1', 'van1', 'su1', 'dia1', 'anh1',
            'toan2', 'ly2', 'hoa2', 'sinh2', 'van2', 'su2', 'dia2', 'anh2']

X = data[features]
y = data['label']

# Chuyển đổi nhãn sang dạng số
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Khởi tạo k-fold cross-validation với k = 5
kfold = KFold(n_splits=5, shuffle=False)

# Tạo danh sách để lưu trữ độ chính xác từ từng fold
accuracies = []

# Lặp qua từng fold và huấn luyện/đánh giá mô hình
for train_index, test_index in kfold.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Xây dựng mô hình MLP
    mlp = MLPClassifier(hidden_layer_sizes=500)
    
    # Huấn luyện mô hình
    mlp.fit(X_train, y_train)
    
    # Đánh giá mô hình trên dữ liệu kiểm tra
    y_pred = mlp.predict(X_test)
    
    # Tính toán độ chính xác và lưu vào danh sách độ chính xác
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# In ra độ chính xác cao nhất từ các fold
max_accuracy = max(accuracies)
print("Độ chính xác cao nhất:", max_accuracy*100, "%")

# Tạo mô hình với độ chính xác cao nhất
best_model_index = accuracies.index(max_accuracy)
best_train_indices, best_test_indices = list(kfold.split(X,y))[best_model_index]
X_train_best, X_test_best = X.iloc[best_train_indices], X.iloc[best_test_indices]
y_train_best, y_test_best = y[best_train_indices], y[best_test_indices]

best_mlp = MLPClassifier(hidden_layer_sizes=500)
best_mlp.fit(X_train_best, y_train_best)


# # Dự đoán
def predict_grade(scores):
    prediction = mlp.predict(scores)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]





# Create the main window
window = tk.Tk()
window.title("Grade Prediction")
window.geometry("1200x800")

# Create input labels and entry fields
tk.Label(window, text="Điểm toán lần 1:").grid(row=0, column=0)
toan1_entry = tk.Entry(window)
toan1_entry.grid(row=0, column=1)

tk.Label(window, text="Điểm lý lần 1:").grid(row=1, column=0)
ly1_entry = tk.Entry(window)
ly1_entry.grid(row=1, column=1)

tk.Label(window, text="Điểm hoa lần 1:").grid(row=2, column=0)
hoa1_entry = tk.Entry(window)
hoa1_entry.grid(row=2, column=1)

tk.Label(window, text="Điểm sinh lần 1:").grid(row=3, column=0)
sinh1_entry = tk.Entry(window)
sinh1_entry.grid(row=3, column=1)

tk.Label(window, text="Điểm van lần 1:").grid(row=4, column=0)
van1_entry = tk.Entry(window)
van1_entry.grid(row=4, column=1)

tk.Label(window, text="Điểm su lần 1:").grid(row=5, column=0)
su1_entry = tk.Entry(window)
su1_entry.grid(row=5, column=1)

tk.Label(window, text="Điểm dia lần 1:").grid(row=6, column=0)
dia1_entry = tk.Entry(window)
dia1_entry.grid(row=6, column=1)

tk.Label(window, text="Điểm anh lần 1:").grid(row=7, column=0)
anh1_entry = tk.Entry(window)
anh1_entry.grid(row=7, column=1)


# ==========


tk.Label(window, text="Điểm toan lần 2:").grid(row=0, column=5)
toan2_entry = tk.Entry(window)
toan2_entry.grid(row=0, column=6)


tk.Label(window, text="Điểm ly lần 2:").grid(row=1, column=5)
ly2_entry = tk.Entry(window)
ly2_entry.grid(row=1, column=6)

tk.Label(window, text="Điểm hoa lần 2:").grid(row=2, column=5)
hoa2_entry = tk.Entry(window)
hoa2_entry.grid(row=2, column=6)

tk.Label(window, text="Điểm sinh lần 2:").grid(row=3, column=5)
sinh2_entry = tk.Entry(window)
sinh2_entry.grid(row=3, column=6)

tk.Label(window, text="Điểm van lần 2:").grid(row=4, column=5)
van2_entry = tk.Entry(window)
van2_entry.grid(row=4, column=6)

tk.Label(window, text="Điểm su lần 2:").grid(row=5, column=5)
su2_entry = tk.Entry(window)
su2_entry.grid(row=5, column=6)

tk.Label(window, text="Điểm dia lần 2:").grid(row=6, column=5)
dia2_entry = tk.Entry(window)
dia2_entry.grid(row=6, column=6)

tk.Label(window, text="Điểm anh lần 2:").grid(row=7, column=5)
anh2_entry = tk.Entry(window)
anh2_entry.grid(row=7, column=6)


# Repeat the above pattern for other input fields

# Create a function to handle the button click event
def predict_grade_getText():
    
    if (toan1_entry.get() == '' or toan2_entry.get() == '' or
        ly1_entry.get() == '' or ly2_entry.get() == '' or
        hoa1_entry.get() == '' or hoa2_entry.get() == '' or
        sinh1_entry.get() == '' or sinh2_entry.get() == '' or
        van1_entry.get() == '' or van2_entry.get() == '' or
        su1_entry.get() == '' or su2_entry.get() == '' or
        dia1_entry.get() == '' or dia2_entry.get() == '' or
        anh1_entry.get() == '' or anh2_entry.get() == ''
        ):
        messagebox.showwarning("Invalid Input", "Diem du lieu khong duoc trong rong.")
        return

    # scrore class 11
    toan1 = float(toan1_entry.get())
    ly1 = float(ly1_entry.get())
    hoa1 = float(hoa1_entry.get())
    sinh1 = float(sinh1_entry.get())
    van1 = float(van1_entry.get())
    su1 = float(su1_entry.get())
    dia1 = float(dia1_entry.get())
    anh1 = float(anh1_entry.get())


    # scrore class 11
    toan2 = float(toan2_entry.get())
    ly2 = float(ly2_entry.get())
    hoa2 = float(hoa2_entry.get())
    sinh2 = float(sinh2_entry.get())
    van2 = float(van2_entry.get())
    su2 = float(su2_entry.get())
    dia2 = float(dia2_entry.get())
    anh2 = float(anh2_entry.get())

    # Check if the input scores are within the valid range
    if not (
        (0 <= toan1 <= 10) or
        (0 <= ly1 <= 10) or 
        (0 <= hoa1 <= 10) or
        (0 <= sinh1 <= 10) or
        (0 <= van1 <= 10) or
        (0 <= su1 <= 10) or
        (0 <= dia1 <= 10) or
        (0 <= anh1 <= 10) or

        (0 <= toan2 <= 10) or
        (0 <= ly2 <= 10) or 
        (0 <= hoa2 <= 10) or
        (0 <= sinh2 <= 10) or
        (0 <= van2 <= 10) or
        (0 <= su2 <= 10) or
        (0 <= dia2 <= 10) or
        (0 <= anh2 <= 10) 
    ):
        messagebox.showwarning("Invalid Input", "Điểm phải nằm trong khoảng từ 0 đến 10.")
        return

    # Perform the grade prediction
    input_scores = [[
        toan1, ly1, hoa1, sinh1, van1, su1, dia1, anh1,
        toan2, ly2, hoa2, sinh2, van2, su2, dia2, anh2
    ]]
    predicted_grade = predict_grade(input_scores)
    result = result_predict(predicted_grade)

    # Show the predicted result
    messagebox.showinfo("Result", f"Kết quả dự đoán: {result}")

# Create the predict button
predict_button = tk.Button(window, text="Dự đoán", command=predict_grade_getText)
predict_button.grid(row=10, column=3, columnspan=2)

# Start the main event loop
window.mainloop()








# # Đánh giá mô hình tốt nhất trên dữ liệu kiểm tra
# y_pred_best = best_mlp.predict(X_test_best)
# classification_best = classification_report(y_test_best, y_pred_best)

# print("Báo cáo phân loại mô hình tốt nhất:")
# print(classification_best)


# # Nhập dữ liệu và dự đoán
# def get_valid_score(message):
#     while True:
#         try:
#             score = float(input(message))
#             if score < 0 or score > 10:
#                 print("Điểm phải nằm trong khoảng từ 0 đến 10. Vui lòng nhập lại.")
#             else:
#                 return score
#         except ValueError:
#             print("Vui lòng nhập một số hợp lệ.")

# toan1 = get_valid_score("Nhập điểm toán lần 1: ")
# ly1 = get_valid_score("Nhập điểm lý lần 1: ")
# hoa1 = get_valid_score("Nhập điểm hoá lần 1: ")
# sinh1 = get_valid_score("Nhập điểm sinh lần 1: ")
# van1 = get_valid_score("Nhập điểm văn lần 1: ")
# su1 = get_valid_score("Nhập điểm sử lần 1: ")
# dia1 = get_valid_score("Nhập điểm địa lần 1: ")
# anh1 = get_valid_score("Nhập điểm anh lần 1: ")
# toan2 = get_valid_score("Nhập điểm toán lần 2: ")
# ly2 = get_valid_score("Nhập điểm lý lần 2: ")
# hoa2 = get_valid_score("Nhập điểm hoá lần 2: ")
# sinh2 = get_valid_score("Nhập điểm sinh lần 2: ")
# van2 = get_valid_score("Nhập điểm văn lần 2: ")
# su2 = get_valid_score("Nhập điểm sử lần 2: ")
# dia2 = get_valid_score("Nhập điểm địa lần 2: ")
# anh2 = get_valid_score("Nhập điểm anh lần 2: ")

# # Tạo dữ liệu đầu vào
# input_scores = [[toan1, ly1, hoa1, sinh1, van1, su1, dia1, anh1,
#                  toan2, ly2, hoa2, sinh2, van2, su2, dia2, anh2]]


# # Dự đoán kết quả
# predicted_grade = predict_grade(input_scores)
# print("Kết quả dự đoán: ", result_predict(predicted_grade))