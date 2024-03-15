import cv2

Ref = cv2.CascadeClassifier("Ref.xml")
camera = cv2.VideoCapture(0)

def face_detection(frame):
    optimize_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = Ref.detectMultiScale(optimize_frame, scaleFactor=1.1, minSize=(100, 100), minNeighbors=5)
    return faces

def face_box(frame):
    for x, y, w, h in face_detection(frame):
        cv2.rectangle(frame, (x,y), (x + w, y + h), (100, 0, ), 4)

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        _, frame = camera.read()
        face_box(frame)
        cv2.imshow("CuyFace AI", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()