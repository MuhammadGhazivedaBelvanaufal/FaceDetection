import cv2

# Load model deteksi wajah
Ref = cv2.CascadeClassifier("Ref.xml")

# Load model gender
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
gender_list = ['Male', 'Female']

# Load model usia
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Kamera
camera = cv2.VideoCapture(0)

# Deteksi wajah
def face_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = Ref.detectMultiScale(gray, scaleFactor=1.1, minSize=(100, 100), minNeighbors=5)
    return faces
# hwahciwha
# Deteksi wajah + jenis kelamin + usia
def face_box(frame):
    faces = face_detection(frame)
    for x, y, w, h in faces:
        face_img = frame[y:y+h, x:x+w].copy()
        
        # Preprocess wajah untuk model DNN
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.426, 87.768, 114.896), swapRB=False)
        
        # Prediksi gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        
        # Prediksi usia
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        
        # Label
        label = f"{gender}, {age}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 255), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Tutup kamera
def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

# Main loop
def main():
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        face_box(frame)
        cv2.imshow("CuyFace AI - Gender & Age Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    close_window()

if __name__ == '__main__':
    main()
