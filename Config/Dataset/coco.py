from .coco_utils import get_coco_api_from_dataset, get_coco, get_coco_kp
import Config.Dataset.transforms as T
import os

def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2),
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

BATCH_SIZE = 4

def load_dataset():
    dataset_test, num_classes = get_dataset(
    "coco", "val", get_transform(), "Config/Dataset/coco"
    )
    return None,dataset_test