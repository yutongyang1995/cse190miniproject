#RANSAC
import numpy as np
import cv2


def ransacForF(matches, minSamples, iterations, threshold, kp1, kp2):
    bestF = None;
    bestNumInliers = 0;
    
    #ensure that at least 8 samples
    if minSamples < 8:
        minsSamples = 8;
        
    for i in range(0, iterations):
        #randomly select inlines
        listOfIndexes = [];
        #possibleInlineMatches = [];
        numInliers = 0

        img1Points = []
        img2Points = []

        counter = 0;
        while counter < minSamples:
            val = random.choice(range(0, len(matches)))
            if val in listOfIndexes:
                continue;

            #possibleInlineMatches.append(matches[val]);
            currentMatch = matches[val]
            img1Points.append(kp1[currentMatch.queryIdx].pt + (1,))
            img2Points.append(kp2[currentMatch.trainIdx].pt + (1,))
            listOfIndexes.append(val);
                
            counter += 1


        f = computeF(img1Points, img2Points)
        #calculated f
        numInliers = 0;
        for match in matches:
          point1 = kp1[match.queryIdx].pt + (1,) 
          point2 = kp2[match.trainIdx].pt + (1,) 

          #print np.linalg.multi_dot((np.array(point1).T, f, point2))

          if abs(np.linalg.multi_dot((np.array(point1).T, f, point2))) < threshold:
            numInliers += 1;

        print numInliers


        if numInliers > bestNumInliers:
          bestF = f;
          bestNumInliers = numInliers;


    
    return bestF;

def computeF(img1Points, img2Points):
    #build constraints
    constraints = [];
    for point1, point2 in zip(img1Points, img2Points):
        x1 = point1[0];
        y1 = point1[1];
        x2 = point2[0];
        y2 = point2[1];
        constraints.append(np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]));
    
    #calculate svd
    U, D, V = np.linalg.svd(constraints)
    
    F = np.reshape(V[8], (3,3))
    
    #ensure rank 2
    U, D, V = np.linalg.svd(F)
    F = np.dot(U, np.dot(np.diag([D[0], D[1], 0]), V))
    
    return F;


file1 = "fox1.tiff"
file2 = "fox2.tiff"

img1 = cv2.imread(file1,0)          # queryImage
img2 = cv2.imread(file2,0) # trainImage


# SIFT
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None)

fig, ax = plt.subplots(figsize=(18, 16), dpi=80)

f = ransacForF(matches, 8, 100, 0.005, kp1, kp2);

newImg = np.zeros(img1.shape);
for y in range(len(img1)):
  for x in range(len(img1[y])):
    currentPoint = (x, y, 1);
    transformed = np.dot(np.array(currentPoint).T, f);
    print transformed

    newPoint = (int(float(transformed[0])/transformed[2]), int(float(transformed[1])/transformed[2]));

    if newPoint[0] in range(len(newImg[y])) and newPoint[1] in range(len(newImg)):
      newImg[newPoint[1]][newPoint[0]] = img1[y][x];

print newImg



#plt.show()

