import cv2


def draw_results(img, boxes, scores, classes, classes_labels, min_score=0.5):

    for (xmin, ymin, xmax, ymax), score, cl in zip(boxes.tolist(),
                                                   scores.tolist(),
                                                   classes.tolist()):
        if score > min_score:
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                          (0, 0, 1), 2)
            text = f'{classes_labels[cl]}: {score:0.2f}'
            cv2.putText(img, text, (int(xmin), int(ymin - 5)),
                        cv2.QT_FONT_NORMAL, 0.5, (0, 0, 1), 1)
    return img