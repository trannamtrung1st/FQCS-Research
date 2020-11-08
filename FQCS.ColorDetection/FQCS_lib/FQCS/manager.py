import numpy as np

COMPARE_FACTOR = 1.5
STEP_FACTOR = 4


class FQCSManager:
    def __init__(self):
        self.__check_group = 0
        self.__last_group_count = 0
        self.__last_check_group = None
        self.__last_check_group_loc = None
        return

    def get_last_check_group_loc(self):
        return self.__last_check_group_loc

    def get_last_group_count(self):
        return self.__last_group_count

    def get_check_group(self):
        return self.__check_group

    def check_group(self):
        self.__last_check_group = self.__check_group
        self.__check_group += 1

    def group_pairs(self, boxes, sample_area):
        max_seperated_area = sample_area * COMPARE_FACTOR if sample_area is not None else None
        grouped = []
        sizes = []
        boxes_count = len(boxes)
        not_sep_count = 0
        for i in reversed(range(0, boxes_count)):
            b = boxes[i]
            c, rect, dimA, dimB, box, tl, tr, br, bl, minx, maxx, cenx = b
            grouped.append([b])
            sizes.append(maxx - minx)
            area = dimA * dimB
            if max_seperated_area is not None and area >= max_seperated_area:
                not_sep_count += 1
                continue
            next_idx = i - 1
            if next_idx > -1:
                next_box = boxes[next_idx]
                next_area = next_box[2] * next_box[3]
                minx = next_box[-3]
                grouped.append([b])
                grouped[-1].append(next_box)
                sizes.append(maxx - minx)

        group_count = len(grouped)
        final_grouped = []
        final_sizes = []
        check_group_loc = None
        if group_count == 0:
            group_count = len(final_grouped)
            self.__calc_check_group(group_count, check_group_loc)
            self.__last_group_count = group_count
            self.__last_check_group_loc = check_group_loc
            return final_grouped, final_sizes
        max_size = np.max(sizes)
        min_size = np.min(sizes)
        range_size = (max_size, max_size)
        if group_count + not_sep_count > 3:
            range_size = self.__devide_range_size(sizes, group_count)
        elif group_count < 3:
            range_size = None
        print("Sizes:", range_size, min_size, max_size)
        print("------------------------")
        for i, g in enumerate(grouped):
            print("Group", i, len(g), sizes[i])
            if (range_size is None or
                (sizes[i] >= range_size[0] and sizes[i] <= range_size[1])
                ) and (self.__last_check_group_loc is None or
                       (len(final_grouped) == self.__last_check_group
                        or g[0][-2] < self.__last_check_group_loc[0])):
                final_grouped.append(g)
                final_sizes.append(sizes[i])

        print("--------- FINAL --------")
        for i, g in enumerate(final_grouped):
            if self.__last_check_group == i:
                right, left = g[0], g[0]
                if len(g) == 2: left = g[1]
                # min_x, max_x
                print("Last:", self.__last_check_group_loc)
                check_group_loc = [left[-3], right[-2]]
                print("Current:", check_group_loc)
            print("Group", i, len(g), final_sizes[i])

        group_count = len(final_grouped)
        self.__calc_check_group(group_count, check_group_loc)
        if check_group_loc is not None:
            self.__last_check_group_loc = check_group_loc
        self.__last_group_count = group_count
        return final_grouped, final_sizes

    def __calc_check_group(self, group_count, check_group_loc):
        diff = self.__last_group_count - group_count
        if diff > 0: self.__check_group -= diff
        elif diff == 0 and (
            (self.__last_check_group_loc is not None
             and check_group_loc is None) or
            (self.__last_check_group_loc is not None
             and check_group_loc is not None
             and self.__last_check_group_loc[0] >= check_group_loc[1])):
            print(self.__last_check_group_loc, check_group_loc)
            self.__check_group -= 1
        self.__check_group = self.__check_group if self.__check_group >= 0 else 0

    def __devide_range_size(self, sizes, group_count):
        sizes = sorted(sizes)
        diffs = []
        max_1, max_2 = 0, 0
        range_1, range_2 = None, None
        for i in range(group_count - 1):
            diff = sizes[i + 1] - sizes[i]
            if diff > max_1:
                max_2 = max_1
                max_1 = diff
                range_2 = range_1
                range_1 = (sizes[i], sizes[i + 1])
            elif diff > max_2:
                max_2 = diff
                range_2 = (sizes[i], sizes[i + 1])
        return (range_2[1], range_1[0])
