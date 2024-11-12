"""
生成三辆车的场景
"""

import csv
import sys

sys.path.append("..")

min_ttc = 1
max_ttc = 1500000


def clear_csv(file_path):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])


def select_action_scenes_except_head(
    inp_csv="../scenelibrary/shortscenes/action_scene_cdm.csv",
):
    with open(inp_csv, "r") as f:
        scene_list = list(csv.reader(f))
    scene_list = [scene_list[i : i + 4] for i in range(0, 400, 4)]
    new_scene_list = []
    for scene in scene_list:
        car1_pos = scene[2][2]
        car2_pos = scene[3][2]
        if not (int(car1_pos) > 0 and int(car2_pos) > 0):
            new_scene_list.append(scene)
    return new_scene_list


if __name__ == "__main__":
    new_csv = "../scenelibrary/shortscenes/action_scene_back.csv"
    new_scenes = select_action_scenes_except_head()
    with open(new_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for scene in new_scenes:
            writer.writerows(scene)
