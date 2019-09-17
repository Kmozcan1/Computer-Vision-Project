import operator
import os
import cv2
import draw_graph
from enum import Enum


# DISPLAY: Displays images and found wheels
# GRAPH: Displays ROC Curve and
# STATS: Show Precision and Recall etc.
class MODE(Enum):
    DISPLAY = 1
    GRAPH = 2
    STATS = 3


# Our cascade classifier
wheel_cascade = cv2.CascadeClassifier('wheel2.xml')

# Change this to change the functionality
current_mode = MODE.STATS

# Threshold. Eliminate the weights below this value
threshold = 2.1

# Global variables
true = []
float_predictions = []
TP = 0
TN = 0
FP = 0
FN = 0
is_a_car = False


# Only take the largest 2 detections if they are above the threshold
def filter_by_weight(wheels):
    global is_a_car
    global largest_weight
    global second_largest_weight

    weights = wheels[2]

    # Take the index and value of the largest weight
    # Then remove it and take the value of the second largest weight
    index1, largest_weight = max(enumerate(weights), key=operator.itemgetter(1))
    weights = list(filter(largest_weight.__ne__, list(weights)))
    index2, second_largest_weight = max(enumerate(weights), key=operator.itemgetter(1))
    if index2 >= index1:
        index2 += 1

    # Is not a car if any of the two largest weights are lower than the threshold
    if largest_weight < threshold or second_largest_weight < threshold:
        is_a_car = False
        best_wheels = [(0, 0, 0, 0), (0, 0, 0, 0)]
    else:
        is_a_car = True
        best_wheels = [wheels[0][index1], wheels[0][index2]]

    # Return the wheels in any case to display the image

    return best_wheels


# Positives Folder
def test_positive_images():
    global float_predictions
    global true
    global TP
    global FN
    global threshold
    global largest_weight
    global second_largest_weight
    global is_a_car

    for image in os.listdir('Positive Images'):
        is_a_car = False
        img = cv2.imread(os.path.join('Positive Images', image))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        largest_weight = 0
        second_largest_weight = 0
        wheels = wheel_cascade.detectMultiScale3(
            gray,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
            outputRejectLevels=True
        )

        # Filter bu weight if there are more than 2 rectangles present
        if len(wheels[0]) >= 2:
            wheels = filter_by_weight(wheels)
        else:
            wheels = [(0, 0, 0, 0), (0, 0, 0, 0)]

        # Display the image
        if current_mode == MODE.DISPLAY:
            for (x, y, w, h) in wheels:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # No need to make the is_a_car check when in GRAPH mode, since the Threshold is zero
        if current_mode == MODE.GRAPH:
            true.append(1)
            float_predictions.append((largest_weight + second_largest_weight) / 2)

        # Increment TP and FN values
        else:
            if is_a_car:
                TP += 1
            else:
                FN += 1


# Negatives Folder
def test_negative_images():
    global float_predictions
    global TN
    global FP
    global true
    global threshold
    global largest_weight
    global second_largest_weight
    global is_a_car

    for image in os.listdir('Negative Images'):
        is_a_car = False
        img = cv2.imread(os.path.join('Negative Images', image))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        largest_weight = 0
        second_largest_weight = 0
        wheels = wheel_cascade.detectMultiScale3(
            gray,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
            outputRejectLevels=True
        )

        # Filter bu weight if there are more than 2 rectangles present
        if len(wheels[0]) >= 2:
            wheels = filter_by_weight(wheels)
        else:
            wheels = [(0, 0, 0, 0), (0, 0, 0, 0)]

        # Display the image
        if current_mode == MODE.DISPLAY:
            for (x, y, w, h) in wheels:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # No need to make the is_a_car check when in GRAPH mode, since the Threshold is zero
        if current_mode == MODE.GRAPH:
            true.append(0)
            float_predictions.append((largest_weight + second_largest_weight) / 2)

        # Increment FP and TN values
        else:
            if is_a_car:
                FP += 1
            else:
                TN += 1


test_positive_images()
test_negative_images()

# Draw graphs if in GRAPH mode
# Show stats in STATS mode
if current_mode == MODE.GRAPH:
    threshold = 0

    # Normalize the weights
    pred_min, pred_max = min(float_predictions), max(float_predictions)
    for i, val in enumerate(float_predictions):
        float_predictions[i] = (val - pred_min) / (pred_max - pred_min)

    draw_graph.roc_curve(true, float_predictions)
    draw_graph.farr(true, float_predictions)
elif current_mode == MODE.STATS:
    print("Calculations for Treshold = " + str(threshold))
    print("True Positives: " + str(TP))
    print("True Negatives: " + str(TN))
    print("False Positives: " + str(FP))
    print("False Negatives: " + str(FN))
    print("*****************************")
    print("Precision: " + str(TP / (TP + FP)))
    print("Recall: " + str(TP / (TP + FN)))
    print("Accuracy: " + str((TP + TN) / (TP + TN + FP + FN)))
    print("F1 Score: " + str(2 * TP / (2 * TP + FP + FN)))
