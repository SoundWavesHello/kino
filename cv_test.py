import numpy as np
import cv2

# so I need a feasible video clip to test on
cap = cv2.VideoCapture('vtest.avi')
count = 0
results = {}
while(cap.isOpened()):
	ret, frame = cap.read()

	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# cv2.imshow('frame',gray)

	# probably edit sample rate
	results[count] = application.get_results(frame)

	count += 1
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

print(results)