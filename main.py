import importlib
from torch.utils.data import DataLoader
from utils import check_device
import numpy as np
import datetime
from Explanation.saliency.saliency_zoo import fast_ig, guided_ig, agi, big, ig, deeplift, saliencymap, sm, sg, process_dataloader, caculate_insert_deletion


class Pipeline:
    def __init__(self):
        self.device = check_device()
        self.dataset = self.load_dataset()
        self.composed_model = self.load_model()
        self.experiment_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def load_dataset(self):
        DATASET = input("Choose dataset: (Default: example)\n") or "example"
        dataset = importlib.import_module(
            f"Config.Dataset.{DATASET}").load_dataset()
        return dataset

    def load_model(self):
        MODEL = input("Choose model: (Default: example)\n") or "example"
        model = importlib.import_module(f"Config.Model.{MODEL}")
        composed_model = model.ComposedModel()
        if model.NEED_NORMALIZE:
            self.data_min = 0
            self.data_max = 1
        else:
            mean = model.MEAN
            std = model.STD
            self.data_min = np.min((0 - np.array(mean)) /
                                   np.array(std))
            self.data_max = np.max((1 - np.array(mean)) /
                                   np.array(std))
        self.num_classes = model.NUM_CLASSES
        self.input_size = model.INPUT_SIZE
        self.hw = np.prod(self.input_size)
        return composed_model

    def run(self):
        while True:
            chosen = input(
                "Choose Module: (1) Adversarial Attack (2) Adversarial Transferability (3) Explanation (4) Basic Indicators (5) Pruning (6) Exit\n")
            if chosen == "3":
                self.run_explanation()
            elif chosen == "4":
                self.run_indicators()
            elif chosen == "6":
                break
            else:
                print("Invalid input.")

    def run_explanation(self):
        BATCH_SIZE = input("Set batch size: (Default: 64)\n") or 64
        BATCH_SIZE = int(BATCH_SIZE)
        BATCH_SAVE = input(
            "Save results every batch in order to avoid memory overflow? (y/n) (Default: n)\n") or "n"
        dataloader = DataLoader(self.dataset, batch_size=BATCH_SIZE)
        explanation_method = input(
            "Choose explanation method: \n" +
            "1) Fast Integrated Gradients \n" +
            "2) Guided Integrated Gradients \n" +  # steps=15
            # epsilon=0.05, max_iter=20, topk=20, num_classes=1000
            "3) Adversarial Gradient Integration \n" +
            # data_min=0, data_max=1, epsilons=[36, 64, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.9 * 255, 1.1 * 255], class_num=1000, gradient_steps=50
            "4) Boundary Attributions \n" +
            "5) Integrated Gradients \n" +  # gradient_steps=50
            "6) DeepLift \n" +
            "7) Saliency Map \n" +
            "8) Saliency Gradient \n" +
            "9) Smooth Gradient \n" +  # stdevs=0.15, gradient_steps=50
            "10) Exit \n"
        )
        methods = [fast_ig, guided_ig, agi, big,
                   ig, deeplift, saliencymap, sm, sg]
        if explanation_method not in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
            print("Invalid input.")
            return
        elif explanation_method == "10":
            return
        else:
            method = methods[int(explanation_method) - 1]
        inputs = list()
        if explanation_method == "2":
            steps = input("Set steps: (Default: 15)\n") or 15
            steps = int(steps)
            inputs.append(steps)
        elif explanation_method == "3":
            epsilon = input("Set epsilon: (Default: 0.05)\n") or 0.05
            epsilon = float(epsilon)
            max_iter = input("Set max_iter: (Default: 20)\n") or 20
            max_iter = int(max_iter)
            topk = input("Set topk: (Default: 20)\n") or 20
            topk = int(topk)
            inputs = [epsilon, max_iter, topk, self.num_classes]
        elif explanation_method == "4":
            epsilons = input(
                "Set epsilons: (Default: 36, 64, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.9 * 255, 1.1 * 255)\n") or "36, 64, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.9 * 255, 1.1 * 255"
            epsilons = epsilons.split(",")
            epsilons = [float(epsilon) for epsilon in epsilons]
            gradient_steps = input("Set gradient_steps: (Default: 50)\n") or 50
            gradient_steps = int(gradient_steps)
            inputs = [self.data_min, self.data_max,
                      epsilons, self.num_classes, gradient_steps]
        elif explanation_method == "5":
            gradient_steps = input("Set gradient_steps: (Default: 50)\n") or 50
            gradient_steps = int(gradient_steps)
            inputs.append(gradient_steps)
        elif explanation_method == "9":
            stdevs = input("Set stdevs: (Default: 0.15)\n") or 0.15
            stdevs = float(stdevs)
            gradient_steps = input("Set gradient_steps: (Default: 50)\n") or 50
            gradient_steps = int(gradient_steps)
            inputs = [stdevs, gradient_steps]
        BATCH_SAVE = BATCH_SAVE == "y"
        results = process_dataloader(
            self.composed_model, dataloader, method, self.experiment_name, BATCH_SAVE, *inputs)
        if (input("Caculate Insertion Deletion Score? (y/n) (Default: y)\n") or "y") == "y":
            caculate_insert_deletion(self.composed_model,self.experiment_name,self.hw,self.num_classes,BATCH_SIZE,results)

    def run_indicators(self):
        BATCH_SIZE = input("Set batch size: (Default: 128)\n") or 128
        BATCH_SIZE = int(BATCH_SIZE)

        dataloader = DataLoader(self.dataset, batch_size=BATCH_SIZE)
        metric_type = input("Choose metric: (1) builtin (2) custom\n") or "1"

        if metric_type == "1":
            compute_metrics = importlib.import_module(
                f"Indicator.builtin").compute_metrics
            compute_metrics(self.composed_model, dataloader,
                            self.experiment_name)
        elif metric_type == "2":
            compute_metrics = importlib.import_module(
                f"Indicator.custom").compute_metrics
            compute_metrics(self.composed_model, dataloader,
                            self.experiment_name)
        else:
            print("Invalid input.")


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
