import cv2

from tracker import EuclideanDistTracker


def main():
    video_path = input("Enter the video path: ")
    cap = cv2.VideoCapture(video_path)
    tracker = EuclideanDistTracker()

    # Cria uma mascara para separar o background dos objetos em movimento
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    while True:
        ret, frame = cap.read()

        # Estraindo a regiao de interesse (ROI)

        height, width, _ = frame.shape
        roi = frame[340:720, 500:800]

        # Aplica a mascara para cada frame
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

        # Busca os contornos retangulares dos objetos em movimento
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            # Remove objetos pequenos do contorno e desenhando um retangulo nos objetos
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)

                detections.append([x, y, w, h])
                boxes_ids = tracker.update(detections)
                for box_id in boxes_ids:
                    x, y, w, h, id = box_id
                    cv2.putText(
                        roi,
                        str(id),
                        (x, y - 15),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (0, 0, 255),
                        2,
                    )
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 3)

        cv2.imshow("FRAME", frame)

        key = cv2.waitKey(1)
        if key == ord("s"):
            break
    cap.release()
    cv2.destroyAllWindows()


if "__main__":
    main()
