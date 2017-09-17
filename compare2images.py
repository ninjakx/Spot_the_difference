import argparse
import cv2
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
import imutils
 
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--image1", required = True, help = "Path to the image")
ap.add_argument("-i2", "--image2", required = True, help = "Path to the image")
args = vars(ap.parse_args())
 
image1 = cv2.imread(args["image1"])
image2 = cv2.imread(args["image2"])
gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

(score,diff) = compare_ssim(gray1,gray2,full=True)
diff = (diff * 255).astype("uint8")
print("SSIM : {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 255), 2)



# show the output images
plt.subplot(121)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

plt.subplot(122)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

plt.show()




