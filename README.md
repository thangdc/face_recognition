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
            
  - Sau đó dùng lệnh sau để tạo ra tập tin huấn luyện (model.pkl)
  
        - python train.py -f "đường dẫn thư mục cha" -c "thuật toán dùng để huấn luyện: (LinearSvm, GridSearchSvm, GMM, RadialSvm, DecisionTree, GaussianNB)"
        
        - Sau khi chạy lệnh này xong bạn sẽ thấy tập tin model_classifier.pkl được tạo trong thư mục train của project
        
   ![Training faces](../master/imgs/face_recognition_training.png)
 
  
+ Nhận diện khuôn mặt của hình ảnh trong thư mục
  
  - Sau khi có được dữ liệu huấn luyện (model_classifier.pkl), chạy dòng lệnh sau (tham số -e không bắt buộc):
  
        - python facerec_from_folder_images.py -f "đường dẫn thư mục chứa ảnh cần nhận diện" -e "đường dẫn thư mục cần lưu khuôn mặt trích xuất"
        
    ![Recognition from folder image](../master/imgs/face_recognition_from_folder_images.png)
    
    ![Recognition from folder image result](../master/imgs/face_recognition_from_folder_images_result.png)
        
+ Nhận diện khuôn mặt trong video:

    - Để nhận diện khuôn mặt trong video chạy dòng lệnh sau (nếu không có tham số -f thì video sẽ được lấy từ webcam, nếu tham số -f có thể là tập tin video hoặc link youtube):
    
          - python facerec_from_videopy -f "đường dẫn đến file video"
          
    ![Recognition from video file](../master/imgs/face_recognition_from_video_file.png)
          
    ![Recognition from video file result](../master/imgs/face_recognition_from_video_file_result.png)


Tham khảo: 

https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78

https://medium.com/@muehler.v/node-js-face-recognition-js-simple-and-robust-face-recognition-using-deep-learning-ea5ba8e852
