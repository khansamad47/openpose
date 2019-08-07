import sys
sys.path.append('../../python');
from openpose import pyopenpose as op
import cv2

params = {
    "model_folder" : "../../../models/",
    #    "face" : True,
    #    "hand" : True
}

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Process Image
image_path="../../../examples/media/COCO_val2014_000000000241.jpg"
imageToProcess = cv2.imread(image_path)

datum = op.Datum()
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop([datum])

# Display Image
print("Body keypoints: \n" + str(datum.poseKeypoints))
print("Face keypoints: \n" + str(datum.faceKeypoints))
print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
print ("Press 0 to exit");
cv2.waitKey(0)
