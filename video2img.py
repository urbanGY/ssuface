import cv2 as cv
cap = cv.VideoCapture('case.mp4')
index = 0
while(cap.isOpened()):
    name = 'test'
    index += 1
    ret, frame = cap.read()
    if frame is None:
        print('end')
        break;
    frame = cv.resize(frame,dsize=(48,48),interpolation=cv.INTER_AREA)
    name = name + str(index)
    cv.imwrite('data/'+name+'.jpg',frame,params=[cv.IMWRITE_JPEG_QUALITY,90])
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == 27:
        break;
cap.release()
cv.destroyAllWindows()

