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


def select_action_scenes_except_ttc(
    inp_csv="../scenelibrary/shortscenes/action_scene_cdm.csv",
):
    with open(inp_csv, "r") as f:
        scene_list = list(csv.reader(f))
    scene_list = [scene_list[i : i + 4] for i in range(0, 400, 4)]
    new_scene_list = []
    for scene in scene_list:
        ego_v = scene[1][0]
        car1_v, car1_lane, car1_pos = scene[2][:3]
        car2_v, car2_lane, car2_pos = scene[3][:3]
        car1_ttc = car2_ttc = -1e9
        if int(car1_lane) == 1:
            car1_ttc = int(car1_pos) / (
                int(ego_v) - int(car1_v) if ego_v != car1_v else 1e-9
            )
        if int(car2_lane) == 1:
            car2_ttc = int(car2_pos) / (
                int(ego_v) - int(car2_v) if ego_v != car2_v else 1e-9
            )
        if (car1_ttc > max_ttc or car1_ttc < 0) and (
            car2_ttc > max_ttc or car2_ttc < 0
        ):
            new_scene_list.append(scene)
    return new_scene_list


if __name__ == "__main__":
    new_csv = "../scenelibrary/shortscenes/action_scene_diff.csv"
    new_scenes = select_action_scenes_except_ttc()
    with open(new_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for scene in new_scenes:
            writer.writerows(scene)
