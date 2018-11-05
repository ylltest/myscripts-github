import cv2
rtsp_url = input("rtsp_url:")
cap = cv2.VideoCapture(rtsp_url)
print(cap.isOpened())
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_path = r'E:\output.avi'
out = cv2.VideoWriter(out_path, fourcc, 20, (768,432))
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        a = out.write(frame)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:break
cap.release()
out.release()
cv2.destroyAllWindows()