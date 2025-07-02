import cv2

# 画像を読み込む
image = cv2.imread('photo.png')

# 顔認識のデータ（OpenCVが用意している）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# グレースケールにする（顔認識に必要）
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 顔を検出
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# 顔に四角を書く
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 結果を表示
cv2.imshow('顔認識', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
