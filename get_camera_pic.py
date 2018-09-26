import cv2
rtsp_url = input("rtsp_url:")
out_path = input("out_path:")
cap = cv2.VideoCapture(rtsp_url)
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow("capture", frame)
    count += 1
    cv2.imwrite(out_path + '\\' + str(count) + '.jpg', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # cv2.imwrite(out_path, frame)
        break
cap.release()
cv2.destroyAllWindows()