import numpy as np

COMPARE_FACTOR = 1.5
STEP_FACTOR = 4


class FQCSManager:
    def __init__(self):
        self.__last_group_count = 0
        self.__last_check_min_x = None
        return

    def get_last_check_min_x(self):
        return self.__last_check_min_x

    def get_last_group_count(self):
        return self.__last_group_count

    def check_group(self, min_x):
        self.__last_check_min_x = min_x

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
        final_status = []
        if group_count == 0:
            group_count = len(final_grouped)
            self.__last_group_count = group_count
            return final_grouped, final_sizes, final_status
        max_size = np.max(sizes)
        min_size = np.min(sizes)
        range_size = (max_size, max_size)
        if group_count < 3:
            range_size = None
        elif group_count + not_sep_count > 3:
            range_size = self.__devide_range_size(sizes, group_count)
        print("Sizes:", range_size, min_size, max_size)
        print("------------------------")
        for i, g in enumerate(grouped):
            print("Group", i, len(g), sizes[i])
            if (range_size is None or
                (sizes[i] >= range_size[0] and sizes[i] <= range_size[1])):
                final_grouped.append(g)
                final_sizes.append(sizes[i])

        print("--------- FINAL --------")
        tmp_last_check_min_x = self.__last_check_min_x
        check_group = None
        for i, g in enumerate(final_grouped):
            status = self.__calc_status(g)
            final_status.append(status)
            if status: tmp_last_check_min_x = self.get_min_x(g)
            elif check_group is None:
                check_group = i
            print("Group", i, len(g), final_sizes[i])
        self.__last_check_min_x = tmp_last_check_min_x

        group_count = len(final_grouped)
        self.__last_group_count = group_count
        return final_grouped, final_sizes, final_status, check_group

    def __calc_status(self, group):
        final_cen_x = self.__get_cen_x(group)
        return self.__last_check_min_x is not None and final_cen_x > self.__last_check_min_x

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
        min_1 = np.min(range_1)
        min_2 = np.min(range_2)
        if min_1 < min_2:
            return (range_1[1], range_2[0])
        return (range_2[1], range_1[0])

    def get_min_x(self, group):
        final_min_x = None
        for b in group:
            c, rect, dimA, dimB, box, tl, tr, br, bl, minx, maxx, cenx = b
            if final_min_x is None or minx < final_min_x:
                final_min_x = minx
        return final_min_x

    def __get_cen_x(self, group):
        final_cen_x = None
        for b in group:
            c, rect, dimA, dimB, box, tl, tr, br, bl, minx, maxx, cenx = b
            if final_cen_x is None or cenx < final_cen_x:
                final_cen_x = cenx
        return final_cen_x