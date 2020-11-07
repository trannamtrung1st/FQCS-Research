import numpy as np

COMPARE_FACTOR = 1.5
MAX_TO_REMOVE = 5


class FQCSManager:
    def __init__(self):
        self.pos_tracks = []
        self.pairs = []
        self.check_group = 0
        self.current_count = 0
        self.removes = {}
        self.last_checked_min_x = None
        self.last_checked_group = None
        return

    def checked_group(self, group):
        self.check_group = group + 1
        self.last_checked_group = group

    def group_pairs(self, boxes, sample_area):
        print("--------------------------")
        for k in self.removes.keys():
            self.removes[k] += 1
        updated_pairs = {}
        updated_pos = {}
        for i in reversed(range(len(boxes))):
            b = boxes[i]
            c, rect, dimA, dimB, box, tl, tr, br, bl, min_x, max_x, center_x = b
            area = dimA * dimB
            finished, group = self.get_group_by_position(center_x)
            if group >= self.current_count:
                # if (self.last_checked_min_x is not None
                #         and center_x > self.last_checked_min_x):
                #     continue
                updated_pairs[group] = [group, b])
                self.current_count += 1
                updated_pos[group] = [group, finished, min_x, max_x]
            elif not finished:
                updated_pos[group][1] = True

            self.removes[group] = 0

            if sample_area is not None and area >= sample_area * COMPARE_FACTOR:
                self.pos_tracks[group][1] = True
                if len(self.pairs[group]) == 3:
                    self.pairs[group][-1] = None

            print("Last:", self.last_checked_min_x)
            print("Box:", center_x, self.pos_tracks[group][1], group)

            
            if is_left or self.is_seperated(group):
                self.pairs[group][2] = b
                self.pos_tracks[group][2] = min_x
            if not is_left or self.is_seperated(group):
                self.pairs[group][1] = b
                self.pos_tracks[group][3] = max_x
            # if group == self.last_checked_group:
            #     last_pos = self.pos_tracks[group]
            #     if last_pos is not None:
            #         self.last_checked_min_x = last_pos[2]
            #     else:
            #         self.last_checked_min_x = None

        deleted = []
        for k in self.removes.keys():
            if self.removes[k] == MAX_TO_REMOVE:
                self.pairs[k] = None
                self.pos_tracks[k] = None
                deleted.append(k)
        for k in deleted:
            # if k == self.last_checked_group:
            #     self.last_checked_group = None
            #     self.last_checked_min_x = None
            del self.removes[k]

    def is_seperated(self, group):
        return len(
            self.pairs[group]) == 2 and self.pos_tracks[group][1] == True

    def get_group_by_position(self, center_x):
        for i in range(self.current_count):
            pos_track = self.pos_tracks[i]
            if pos_track is None: continue
            group, finished, min_x, max_x = pos_track
            is_in = min_x <= center_x and max_x >= center_x
            center_group = (min_x + max_x) / 2
            print("Group: ", group, min_x, max_x)
            if is_in or not finished:
                return (finished, group)
        return (False, self.current_count, False)
