import numpy as np

COMPARE_FACTOR = 1.5
MAX_TO_REMOVE = 5


class FQCSManager:
    def __init__(self):
        return

    def group_pairs(self, boxes):
        grouped = []
        sizes = []
        for i in reversed(range(0, len(boxes))):
            b = boxes[i]
            c, rect, dimA, dimB, box, tl, tr, br, bl, minx, maxx, cenx = b
            grouped.append([b])
            sizes.append(maxx - minx)
            next_idx = i - 1
            if next_idx > -1:
                grouped.append([b])
                grouped[-1].append(boxes[next_idx])
                minx = boxes[next_idx][-3]
                sizes.append(maxx - minx)
        for i, g in enumerate(grouped):
            print("Group", i, len(g), sizes[i])
        return
