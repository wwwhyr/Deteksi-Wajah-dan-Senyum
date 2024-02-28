import cv2
import time

# Inisialisasi Video
smilevideo = cv2.VideoCapture(0)
smilevideo.set(3, 640)  # Lebar
smilevideo.set(4, 480)  # Tinggi

# Menerapkan data wajah dan senyum
smile_data = cv2.CascadeClassifier('HaarcascadeFiles/Smile.xml')
face_data = cv2.CascadeClassifier('HaarcascadeFiles/haarcascade_frontalface_default.xml')


img_counter = 0


smile_counter = 0
smile_stable_duration = 20

while True:
    # Read frame from video capture
    read_successful, frame = smilevideo.read()

    if read_successful:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah
        face_coords = face_data.detectMultiScale(gray_frame)

        for cell in face_coords:
            x, y, width, height = cell
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            FaceText = cv2.putText(
                frame, 'Face', (x, y - 10), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            croped_face = frame[y:y + height, x:x + width]
            grayscaled_face = cv2.cvtColor(croped_face, cv2.COLOR_BGR2GRAY)

            # Deteksi senyum
            smile_coords = smile_data.detectMultiScale(grayscaled_face, scaleFactor=1.7, minNeighbors=20)

            for cell in smile_coords:
                a, b, c, d = cell
                cv2.rectangle(croped_face, (a, b),
                              (a + c, b + d), (0, 255, 0), 2)

                SmileText = cv2.putText(
                    croped_face, 'Smile', (a, b - 10), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


                # Mempotret otomatis ketika senyum terdeteksi
                smile_counter += 1

                if smile_counter >= smile_stable_duration:
                    
                    img_name = "smile_detected_{}.png".format(img_counter)
                    cv2.imwrite(img_name, frame)
                    print("{} written!".format(img_name))
                    img_counter += 1

                    
                    smile_counter = 0

        # Nama frame
        cv2.imshow("Deteksi Senyum", frame)

        
        key = cv2.waitKey(1)

        # Tekan q/Q untuk mematikan video
        if key == 81 or key == 113:
            break
    else:
        break


smilevideo.release()
cv2.destroyAllWindows()
