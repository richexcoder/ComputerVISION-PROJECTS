# import cv2
# import easyocr
# import matplotlib.pyplot as plt
#
# # start camera
# cap = cv2.VideoCapture(0)
#
# # instance text detectorq
# reader = easyocr.Reader(['en'], gpu=False)
#
# while True:
#     ret, img = cap.read()
#
#     if not ret:
#         break
#
#     # detect text on image
#     text_ = reader.readtext(img)
#
#     # draw bbox and text
#     for t in text_:
#         print(t)
#
#         bbox, text, score = t
#
#         cv2.rectangle(img, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
#         cv2.putText(img, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)
#
#     # show result (replaces matplotlib)
#     cv2.imshow("Image", img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


