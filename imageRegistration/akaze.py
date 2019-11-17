import cv2

img1 = cv2.imread('too.png')
img2 = cv2.imread('too2.png')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

akaze = cv2.AKAZE_create()

kp1, des1 = akaze.detectAndCompute(gray1, None)
kp2, des2 = akaze.detectAndCompute(gray2, None)

# save AKAZE feature registration
output=cv2.drawKeypoints(gray1, kp1, img1)
cv2.imwrite('akaze_kp_descriptor.png', output)

# use K-Nearest Neighbour (k=2) to match 
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_matches = list()
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append([m])
    
output2 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('knn_matches.png', output2)
