import matplotlib.pyplot as plt


# To draw graphs

def regressor_to_classifier(predictions, threshold=0.5):
    output = []
    for prediction in predictions:
        if threshold == 0:
            output.append(1)
        elif threshold == 0.99:
            output.append(0)
        elif prediction > threshold:
            output.append(1)
        else:
            output.append(0)
    return output


def confusion_matrix(true, predictions):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for t, p in zip(true, predictions):
        if t == 1 and p == 1:
            TP += 1
        elif t == 0 and p == 1:
            FP += 1
        elif t == 1 and p == 0:
            FN += 1
        else:
            TN += 1
    return TP, FP, TN, FN


def roc_curve(true, float_predictions):
    x = []
    y = []
    for i in range(100):
        threshold = 0.01 * i
        bool_predictions = regressor_to_classifier(float_predictions, threshold)
        TP, FP, TN, FN = confusion_matrix(true, bool_predictions)
        if TP + FN != 0:
            TPR = TP / (TP + FN)
        else:
            TRP = 0
        if FP + TN != 0:
            FPR = FP / (FP + TN)
        else:
            FPR = 0
        x.append(FPR)
        y.append(TPR)

    plt.plot(x, y)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()


def farr(true, float_predictions):
    x = []
    y = []
    z = []
    for i in range(100):
        threshold = 0.01 * i
        z.append(threshold)
        bool_predictions = regressor_to_classifier(float_predictions, threshold)
        print("Threshold = {}".format(threshold))
        TP, FP, TN, FN = confusion_matrix(true, bool_predictions)
        if TP + FN != 0:
            FAR = FP / (FP + TN)
        if FP + TN != 0:
            FRR = FN / (TP + FN)
        x.append(FAR)
        y.append(FRR)

    plt.plot(z, x, label="FAR")
    plt.plot(z, y, label="FRR")
    plt.xlabel("Threshold")
    plt.ylabel("FAR and FRR")
    plt.title("FAR and FRR")
    plt.legend()
    plt.show()
