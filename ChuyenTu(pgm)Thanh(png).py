from PIL import Image
import os

# Đường dẫn đến thư mục chứa hình ảnh .pgm
pgm_folder_path = "file_path"

# Đường dẫn đến thư mục để lưu trữ hình ảnh .png đã chuyển đổi
png_folder_path = 'file_path'

# Tạo thư mục nếu nó không tồn tại
os.makedirs(png_folder_path, exist_ok=True)

# Lặp qua tất cả các file trong thư mục .pgm
for filename in os.listdir(pgm_folder_path):
    if filename.endswith('.pgm'):
        # Đường dẫn đầy đủ đến file .pgm
        pgm_file_path = os.path.join(pgm_folder_path, filename)

        # Đường dẫn đến file .png sau khi chuyển đổi
        png_file_path = os.path.join(png_folder_path, os.path.splitext(filename)[0] + '.png')

        # Mở hình ảnh .pgm
        with Image.open(pgm_file_path) as img:
            # Lưu hình ảnh dưới dạng .png
            img.save(png_file_path, 'PNG')

print("Chuyển đổi hoàn thành!")