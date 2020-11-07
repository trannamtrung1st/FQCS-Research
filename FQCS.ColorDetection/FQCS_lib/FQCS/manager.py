import numpy as np

COMPARE_FACTOR = 1.5


class FQCSManager:
    def __init__(self):
        self.__check_group = 0
        self.__last_group_count = 0
        self.__check_group_loc = None
        return

    def get_last_group_count(self):
        return self.__last_group_count

    def get_check_group(self):
        return self.__check_group

    def check_group(self):
        self.__check_group += 1

    def group_pairs(self, boxes, sample_area):
        max_seperated_area = sample_area * COMPARE_FACTOR if sample_area is not None else None
        grouped = []
        sizes = []
        boxes_count = len(boxes)
        for i in reversed(range(0, boxes_count)):
            b = boxes[i]
            c, rect, dimA, dimB, box, tl, tr, br, bl, minx, maxx, cenx = b
            grouped.append([b])
            sizes.append(maxx - minx)
            area = dimA * dimB
            if max_seperated_area is not None and area >= max_seperated_area:
                continue
            next_idx = i - 1
            if next_idx > -1:
                next_box = boxes[next_idx]
                next_dimA, next_dimB = next_box[2:4]
                next_area = next_dimA * next_dimB
                if max_seperated_area is not None and next_area >= max_seperated_area:
                    continue
                grouped.append([b])
                grouped[-1].append(boxes[next_idx])
                minx = boxes[next_idx][-3]
                sizes.append(maxx - minx)

        group_count = len(grouped)
        final_grouped = []
        final_sizes = []
        if group_count == 0:
            group_count = len(final_grouped)
            self.__calc_check_group(group_count, None)
            self.__last_group_count = group_count
            return final_grouped, final_sizes
        max_size = np.max(sizes)
        min_size = np.min(sizes)
        range_size = (max_size, max_size)
        if group_count > 3:
            step = (max_size - min_size) / 4
            range_size = (min_size + step, max_size - step)
        print("Sizes:", range_size, min_size, max_size)
        print("------------------------")
        for i, g in enumerate(grouped):
            print("Group", i, len(g), sizes[i])
            if sizes[i] >= range_size[0] and sizes[i] <= range_size[1]:
                final_grouped.append(g)
                final_sizes.append(sizes[i])
                if self.__check_group == i:
                    right, left = g[0], g[0]
                    if len(g) == 2: left = g[1]
                    # min_x, max_x
                    self.__check_group_loc = [g[1][-3], g[0][-2]]
        print("--------- FINAL --------")
        for i, g in enumerate(final_grouped):
            print("Group", i, len(g), final_sizes[i])
        group_count = len(final_grouped)
        self.__calc_check_group(group_count)
        self.__last_group_count = group_count
        return final_grouped, final_sizes

    def __calc_check_group(self, group_count, check_group_loc):
        diff = self.__last_group_count - group_count
        if diff > 0: self.__check_group -= diff
        self.__check_group = self.__check_group if self.__check_group >= 0 else 0
