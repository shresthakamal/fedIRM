from operator import index
import pandas as pd
import os
import shutil

root_dir = "dataset/stage_1_train_images"

csv_file = "dataset/stage_1_train.csv"


def truncate_data(TOTAL_IMAGES):

    labels = pd.read_csv("dataset/stage_1_train.csv")

    df = labels[: TOTAL_IMAGES * 6]

    df.to_csv("dataset/truncated_stage_1_train.csv", index=False)

    for i in range(TOTAL_IMAGES * 6):
        items = df.iloc[i]["ID"]

        image_file = items.split("_")[0] + "_" + items.split("_")[1] + ".png"

        image_name = os.path.join(root_dir, image_file)

        destn = os.path.join("dataset/truncated_data/stage_1_train_images", image_file)
        os.makedirs(os.path.dirname(destn), exist_ok=True)

        shutil.copyfile(image_name, destn)


truncate_data(1000)
