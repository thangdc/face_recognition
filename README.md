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
        
   ![Training faces](../master/imgs/face_recognition_training.png)
        
+ Nhận diện khuôn mặt của hình ảnh trong thư mục
  
  - Sau khi có được dữ liệu huấn luyện (model.json), chạy dòng lệnh sau:
  
        - python facerec_from_folder_images.py -f "đường dẫn thư mục chứa ảnh cần nhận diện" -m "đường dẫn của tập tin model.json"
        
    ![Recognition from folder image](../master/imgs/face_recognition_from_folder_images.png)
    
    ![Recognition from folder image result](../master/imgs/face_recognition_from_folder_images_result.png)
        
+ Nhận diện khuôn mặt trong video:

    - Để nhận diện khuôn mặt trong video chạy dòng lệnh sau:
    
          - python facerec_from_video_file.py -f "đường dẫn đến file video" -m "đường dẫn tập tin model.json"
          
    ![Recognition from video file](../master/imgs/face_recognition_from_video_file.png)
          
    ![Recognition from video file result](../master/imgs/face_recognition_from_video_file_result.png)
          
Tham khảo: 

https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78

https://medium.com/@muehler.v/node-js-face-recognition-js-simple-and-robust-face-recognition-using-deep-learning-ea5ba8e852
