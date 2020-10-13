import cv2


def draw_results(images,
                 boxes,
                 scores,
                 classes,
                 classes_labels,
                 img_size,
                 min_score=0.5):

    for i in range(len(images)):
        iboxes = boxes[i] * [
            img_size[0], img_size[1], img_size[0], img_size[1]
        ]
        iscores = scores[i]
        iclasses = classes[i].astype(int)
        for (xmin, ymin, xmax, ymax), score, cl in zip(iboxes.tolist(),
                                                       iscores.tolist(),
                                                       iclasses.tolist()):
            if score > min_score:
                cv2.rectangle(images[i], (int(xmin), int(ymin)),
                              (int(xmax), int(ymax)), (0, 0, 1), 2)
                text = f'{classes_labels[cl]}: {score:0.2f}'
                cv2.putText(images[i], text, (int(xmin), int(ymin - 5)),
                            cv2.QT_FONT_NORMAL, 0.5, (0, 0, 1), 1)
    return images