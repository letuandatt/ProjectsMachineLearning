# import matplotlib.pyplot as plt
# import pandas as pd
#
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.datasets import load_iris
#
# # Load dữ liệu iris
# iris = load_iris()
# X, y = iris.data, iris.target
#
# # Chia dữ liệu thành tập huấn luyện và tập kiểm thử
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Định nghĩa các giá trị tham số cần thử nghiệm
# param_grid = {
#     'hidden_layer_sizes': [(10,), (20,), (30,)],
#     'activation': ['relu', 'tanh'],
#     'solver': ['lbfgs', 'adam'],
#     'max_iter': [500, 1000, 1500],
# }
#
# # Tạo KNNClassifier
# knn = KNeighborsClassifier(n_neighbors=7)
#
# param_grid = {
#     'n_neighbors' : [5, 7, 9, 11, 13]
# }
#
# # Tạo GridSearchCV
# grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
#
# # Tiến hành tìm kiếm bộ tham số tối ưu
# grid_search.fit(X_train, y_train)
#
# # Lấy kết quả và chuyển đổi thành DataFrame
# results = pd.DataFrame(grid_search.cv_results_)
#
# # Biểu đồ tương quan giữa các tham số và độ chính xác
# plt.figure(figsize=(15, 10))
#
# for i, param in enumerate(param_grid):
#     plt.subplot(2, 2, i + 1)
#     ax = plt.gca()
#     grouped_data = results.groupby([f"param_{param}"])['mean_test_score'].mean()
#     bars = grouped_data.plot(kind='bar', color='skyblue')
#
#     plt.title(f'Tuning: {param}')
#     plt.xlabel(param)
#     plt.ylabel('Mean Accuracy')
#
#     for bar in bars.patches:
#         # Hiển thị số trên mỗi cột bar
#         ax.annotate(f'{bar.get_height():.2f}',
#                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
#                     ha='center', va='center',
#                     xytext=(0, 5),
#                     textcoords='offset points')
#
# plt.tight_layout()
# plt.show()
#
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# # Thiết lập đường dẫn đến thư mục chứa dữ liệu
# train_data_dir = 'path/to/training_data'
# test_data_dir = 'path/to/testing_data'
#
# # Thiết lập các tham số
# img_width, img_height = 150, 150
# batch_size = 32
# epochs = 10
#
# # Sử dụng ImageDataGenerator để tạo các bộ tăng cường dữ liệu
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
#
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# # Tạo các generators cho tập huấn luyện và kiểm tra
# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary'
# )
#
# test_generator = test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary'
# )
#
# # Xây dựng mô hình CNN đơn giản
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# # Biên dịch mô hình
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Huấn luyện mô hình
# model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     epochs=epochs,
#     validation_data=test_generator,
#     validation_steps=test_generator.samples // batch_size
# )