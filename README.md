# Nhận diện khuôn mặt

1. Hướng dẫn cài đặt OpenCV 3 và Dlib trên Windows
- Xem chi tiết tại thư mục: docs/Install OpenCV 3 and Dlib on Windows.docx

2. Hướng dẫn sử dụng

+ Huấn luyện dữ liệu:

  - Tạo ra thư mục theo cấu trúc sau:
  
        - Thư mục cha
            - Tên (Nguyen Van A)
                - Hình (1.jpg)
                - Hình (2.jpg)
                - ...
            - Tên (Nguyen Van B)
                - Hình (1.jpg)
                - Hình (2.jpg)
                - ...
            - ...
            
  - Sau đó dùng lệnh sau để tạo ra tập tin huấn luyện (model.json)
  
        - python train.py -f "đường dẫn thư mục cha"
        
        - Sau khi chạy lệnh này xong bạn sẽ thấy tập tin model.json được tạo trong "đường dẫn thư mục cha"
        
+ Nhận diện khuôn mặt của hình ảnh trong thư mục
  
  - Sau khi có được dữ liệu huấn luyện (model.json), chạy dòng lệnh sau:
  
        - python facerec_from_folder_images.py -f "đường dẫn thư mục chứa ảnh cần nhận diện" -m "đường dẫn của tập tin model.json"
        
+ Nhận diện khuôn mặt trong video:

    - Để nhận diện khuôn mặt trong video chạy dòng lệnh sau:
    
          - python facerec_from_video_file.py -f "đường dẫn đến file video" -m "đường dẫn tập tin model.json"
