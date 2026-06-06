import json
import os
import random
from collections import defaultdict
from typing import List

from stamo.renderer.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


random.seed(33)


def create_split_jsonl(train_image_dirs: List[str], eval_image_dirs: List[str], dataset_name: str):
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    def collect_images(dirs):
        dir_to_images = defaultdict(list)
        for image_dir in dirs:
            for root, _, files in os.walk(image_dir):
                for file in files:
                    if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                        abs_path = os.path.abspath(os.path.join(root, file))
                        dir_to_images[image_dir].append(abs_path)
            overwatch.info(f"Collected {len(dir_to_images[image_dir])} images from {image_dir}")
        return dir_to_images

    # 收集所有图片
    train_dir_to_images = collect_images(train_image_dirs)
    eval_dir_to_images = collect_images(eval_image_dirs)

    overwatch.info(
        f"Collection completed: {sum(len(v) for v in train_dir_to_images.values())} train images, "
        f"{sum(len(v) for v in eval_dir_to_images.values())} eval images in total."
    )

    os.makedirs("./jsons", exist_ok=True)

    eval_images_entries = []
    cnt = 0
    # 处理 train_image_dirs：全部写入 train.jsonl，每个目录抽取 3 张写入 eval.jsonl
    for image_dir, images in train_dir_to_images.items():
        train_images_set = set()
        if len(images) < 3:
            raise ValueError(f"训练目录 {image_dir} 中图像不足 3，无法抽取 eval 用图像。")

        # 生成名字后缀（取目录名）
        suffix = f"part_{cnt}"
        train_jsonl_path = f"./jsons/train_{dataset_name}_{suffix}.jsonl"
        cnt += 1

        train_images_set.update(images)
        shared_for_eval = random.sample(images, 3)
        for img in shared_for_eval:
            eval_images_entries.append({"image": img, "//": "from-train-shared"})

        with open(train_jsonl_path, "w", encoding="utf-8") as f_train:
            for img in sorted(train_images_set):
                f_train.write(json.dumps({"image": img}) + "\n")

        overwatch.info(
            f"Train part {suffix}: wrote {len(train_images_set)} images to {train_jsonl_path}, 3 shared with eval."
        )

    # 处理 eval_image_dirs：每个目录抽取 3 张 eval-only 图像
    for image_dir, images in eval_dir_to_images.items():
        if len(images) < 3:
            raise ValueError(f"评估目录 {image_dir} 中图像不足 3，无法抽取 eval 用图像。")
        eval_only = random.sample(images, 3)
        for img in eval_only:
            eval_images_entries.append({"image": img, "//": "from-eval-only"})

    eval_jsonl_path = f"./jsons/eval_{dataset_name}.jsonl"
    with open(eval_jsonl_path, "w", encoding="utf-8") as f_eval:
        for entry in eval_images_entries:
            f_eval.write(json.dumps(entry) + "\n")

    overwatch.info(f"Eval: {len(eval_images_entries)} images written to {eval_jsonl_path}")

    datasets = [f"train_{dataset_name}_part_{i}.jsonl" for i in range(cnt)]
    ratios = [1 / cnt for _ in range(cnt)]
    train_info = {"datasets": datasets, "ratios": ratios}
    train_json_path = f"./jsons/train_{dataset_name}.json"
    with open(train_json_path, "w", encoding="utf-8") as f_train:
        json.dump(train_info, f_train, indent=4)
    overwatch.info(f"Train config written to {train_json_path}, {cnt} parts.")

    datasets = [f"eval_{dataset_name}.jsonl"]
    ratios = [1]
    eval_info = {"datasets": datasets, "ratios": ratios}
    eval_json_path = f"./jsons/eval_{dataset_name}.json"
    with open(eval_json_path, "w", encoding="utf-8") as f_eval:
        json.dump(eval_info, f_eval, indent=4)
    overwatch.info(f"Eval config written to {eval_json_path}")


if __name__ == "__main__":
    train_image_dirs = [
        "/mnt/public/dataset2/RoboRender_datasets/libero/libero_10/train",
        "/mnt/public/dataset2/RoboRender_datasets/libero/libero_90/train",
        "/mnt/public/dataset2/RoboRender_datasets/libero/libero_goal/train",
        "/mnt/public/dataset2/RoboRender_datasets/libero/libero_object/train",
        "/mnt/public/dataset2/RoboRender_datasets/libero/libero_spatial/train",
        "/mnt/public/dataset2/RoboRender_datasets/droid/AUTOLab",
        "/mnt/public/dataset2/RoboRender_datasets/droid/CLVR",
        "/mnt/public/dataset2/RoboRender_datasets/droid/GuptaLab",
        "/mnt/public/dataset2/RoboRender_datasets/droid/ILIAD",
        "/mnt/public/dataset2/RoboRender_datasets/droid/PennPAL",
        "/mnt/public/dataset2/RoboRender_datasets/droid/RAD",
        "/mnt/public/dataset2/RoboRender_datasets/droid/RAIL",
        "/mnt/public/dataset2/RoboRender_datasets/droid/REAL",
        "/mnt/public/dataset2/RoboRender_datasets/droid/RPL",
        "/mnt/public/dataset2/RoboRender_datasets/droid/TRI",
        "/mnt/public/dataset2/RoboRender_datasets/droid/WEIRD",
    ]

    eval_image_dirs = [
        "/mnt/public/dataset2/RoboRender_datasets/libero/libero_10/eval",
        "/mnt/public/dataset2/RoboRender_datasets/libero/libero_90/eval",
        "/mnt/public/dataset2/RoboRender_datasets/libero/libero_goal/eval",
        "/mnt/public/dataset2/RoboRender_datasets/libero/libero_object/eval",
        "/mnt/public/dataset2/RoboRender_datasets/libero/libero_spatial/eval",
        "/mnt/public/dataset2/RoboRender_datasets/droid/test",
    ]
    create_split_jsonl(train_image_dirs, eval_image_dirs, "VLA")
