
import toml
import importlib
from torch.utils.data import DataLoader
import datetime
from PIL import Image
from utils import check_device
import numpy as np
import glob
import os
from inspect import isfunction
import torch
from Explanation.evaluation.tools import CausalMetric
device = check_device()
experiment_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
import pickle
import argparse
import pickle
import numpy as np
import torch
import random
from tqdm import tqdm
import copy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(2023)

argparser = argparse.ArgumentParser()
argparser.add_argument("--pipeline_config",type=str,default="Config/pipeline.toml")
pipeline_config = toml.load(argparser.parse_args().pipeline_config)
adversarial_config = toml.load("Config/adversarial.toml")
explanation_config = toml.load("Config/explanation.toml")
mutants_config = toml.load("Config/mutants.toml")
pruning_config = toml.load("Config/pruning.toml")
# pipeline_config = toml.load("Config/pipeline.toml")

def eval_config(config):
    for key in config.keys():
        if isinstance(config[key], dict):
            eval_config(config[key])
        else:
            try:
                # if isfunction(eval(config[key])):
                if key != "name":
                    config[key] = eval(config[key])
            except:
                pass
    return config

def apply_mask(model,mask):
    model = copy.deepcopy(model)
    for name, param in model.named_parameters():
        if name in mask:
            param.data = param.data * mask[name]
    return model

adversarial_config = eval_config(adversarial_config)
explanation_config = eval_config(explanation_config)
mutants_config = eval_config(mutants_config)
pruning_config = eval_config(pruning_config)
pipeline_config = eval_config(pipeline_config)

model_name = pipeline_config["Config"]["model"]
if pipeline_config.get("Config").get("name"):
    experiment_name = pipeline_config["Config"]["name"]
    
if pipeline_config.get("Config").get("mask"):
    mask = pickle.load(open(pipeline_config["Config"]["mask"],"rb"))
    if "greg" in pipeline_config["Config"]["mask"]:
        for k in list(mask.keys()):
            mask[k+".weight"] = mask[k]
            del mask[k]

experiment_name = f"{experiment_name}_{model_name}"

model = importlib.import_module(f'Config.Model.{pipeline_config["Config"]["model"]}')
dataset = importlib.import_module(f'Config.Dataset.{pipeline_config["Config"]["dataset"]}')

TYPE = model.TYPE
if TYPE == "Image Classification":
    NUM_CLASSES = model.NUM_CLASSES
    INPUT_SIZE = model.INPUT_SIZE
elif TYPE == "Text Classification":
    NUM_CLASSES = model.NUM_CLASSES
ComposedModel = model.ComposedModel

model = ComposedModel().get_model()
if pipeline_config.get("Config").get("mask"):
    model = apply_mask(model,mask)

train_dataset,test_dataset = dataset.load_dataset()

BATCH_SIZE = dataset.BATCH_SIZE

if train_dataset is not None:
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
if TYPE == "Object Detection":
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,collate_fn=lambda batch: tuple(zip(*batch)))

del pipeline_config['Config']

def mutants_runner(module,f,function,cfg):
    input_ = globals()[cfg["input"]]
    function_config = mutants_config.get(f,{})
    function = function(**function_config)
    globals()[cfg["output"]] = DataLoader(input_.dataset,batch_size=BATCH_SIZE,shuffle=False,collate_fn=function)
    
    
def adversarial_runner(module,f,function,cfg):
    input_ = globals()[cfg["input"]]
    save = cfg["save"]
    function_config = adversarial_config.get(f,{})
    function = function(model,**function_config)
    if f == "advgan":
        function.train(train_dataloader)
    file_name = 0
    all_adversarial_labels = list()
    all_labels = list()
    correct_index = list()
    if cfg.get("other_models",False):
        other_models_predictions = dict()
        for other_model_name in cfg["other_models"].split(","):
            other_model_name = other_model_name.strip()
            other_model = importlib.import_module(f'Config.Model.{other_model_name}')
            globals()[other_model_name] = other_model.ComposedModel().get_model()
            other_models_predictions[other_model_name] = dict()
            other_models_predictions[other_model_name]["all_adversarial_labels"] = list()
            other_models_predictions[other_model_name]["correct_index"] = list()
    for x,y in tqdm(input_):
        x = x.to(device)
        y = y.to(device)
        adversarial_images = function(x,y)
        with torch.no_grad():
            correct_idx = model(x).argmax(-1) == y
            correct_index.append(correct_idx.cpu().detach().numpy())
            adverserial_labels = model(adversarial_images).argmax(-1).cpu().detach().numpy()
            all_labels.append(y.cpu().detach().numpy())
            if cfg.get("other_models",False):
                for other_model_name in cfg["other_models"].split(","):
                    other_model_name = other_model_name.strip()
                    correct_idx = globals()[other_model_name](x).argmax(-1) == y
                    other_models_predictions[other_model_name]["correct_index"].append(correct_idx.cpu().detach().numpy())
                    adversarial_images_tmp = adversarial_images.clone().detach().cpu().numpy()
                    adversarial_images_tmp = adversarial_images_tmp * 255
                    adversarial_images_tmp = adversarial_images_tmp.astype(np.uint8)
                    adversarial_images_tmp = adversarial_images_tmp / 255
                    adversarial_images_tmp = torch.from_numpy(adversarial_images_tmp).float().to(device)
                    other_models_predictions[other_model_name]["all_adversarial_labels"].append(globals()[other_model_name](adversarial_images_tmp).argmax(-1).cpu().detach().numpy())
            if save:
                for adversarial_image in adversarial_images:
                    adversarial_image = adversarial_image.cpu().detach().numpy().transpose(1,2,0)
                    adversarial_image = Image.fromarray((adversarial_image * 255).astype(np.uint8))
                    if not os.path.exists(f"Outputs/{experiment_name}/{module}/{f}"):
                        os.makedirs(f"Outputs/{experiment_name}/{module}/{f}")
                        os.makedirs(f"Outputs/{experiment_name}/{module}/{f}/adv_images")
                    adversarial_image.save(f"Outputs/{experiment_name}/{module}/{f}/adv_images/{file_name}.png")
                    file_name += 1
            all_adversarial_labels.append(adverserial_labels)
    all_adversarial_labels = np.concatenate(all_adversarial_labels)
    all_labels = np.concatenate(all_labels)
    correct_index = np.concatenate(correct_index)
    success_rate = (1 - np.sum(all_adversarial_labels[correct_index] == all_labels[correct_index]) / np.sum(correct_index))
    print(f"{f} Success Rate: {success_rate * 100}%")
    if save:
        result = {
            "all_adversarial_labels":all_adversarial_labels,
            "all_labels":all_labels,
            "success_rate":success_rate
        }
        with open(f"Outputs/{experiment_name}/{module}/{f}/result.pkl","wb") as f1:
            pickle.dump(result,f1)
        with open(f"Outputs/{experiment_name}/{module}/{f}/result.txt","w") as f2:
            f2.write(f"{model_name} Success Rate: {success_rate * 100}%")
            if cfg.get("other_models",False):
                for other_model_name in cfg["other_models"].split(","):
                    other_model_name = other_model_name.strip()
                    other_models_predictions[other_model_name]["all_adversarial_labels"] = np.concatenate(other_models_predictions[other_model_name]["all_adversarial_labels"])
                    other_models_predictions[other_model_name]["correct_index"] = np.concatenate(other_models_predictions[other_model_name]["correct_index"])
                    other_models_predictions[other_model_name]["success_rate"] = (1 - np.sum(other_models_predictions[other_model_name]["all_adversarial_labels"][other_models_predictions[other_model_name]["correct_index"]] == all_labels[other_models_predictions[other_model_name]["correct_index"]]) / np.sum(other_models_predictions[other_model_name]["correct_index"]))
                    f2.write(f"\n{other_model_name} Success Rate: {other_models_predictions[other_model_name]['success_rate'] * 100}%")
                    with open(f"Outputs/{experiment_name}/{module}/{f}/result_{other_model_name}.pkl","wb") as f3:
                        pickle.dump(other_models_predictions[other_model_name],f3)
    
def explanation_runner(module,f,function,cfg):
    input_ = globals()[cfg["input"]]
    save = cfg["save"]
    function_config = explanation_config.get(f,{})
    if f == "big":
        input_ = DataLoader(input_.dataset,batch_size=32,shuffle=False)
    for i,(x,y) in tqdm(enumerate(input_),total=len(input_)):
        x = x.to(device)
        y = y.to(device)
        if f == "fast_ig" or f == "guided_ig" or f == "ig" or f == "sg":
            attribution = list()
            for x_,y_ in tqdm(zip(x,y),total=len(x)):
                x_ = x_.unsqueeze(0)
                y_ = y_.unsqueeze(0)
                attribution_ = function(model,x_,y_,**function_config)
                attribution.append(attribution_)
            attribution = np.concatenate(attribution,axis=0)
        else:
            attribution = function(model,x,y,**function_config)
        if save:
            if not os.path.exists(f"Outputs/{experiment_name}/{module}/{f}"):
                os.makedirs(f"Outputs/{experiment_name}/{module}/{f}")
            y = y.cpu().detach().numpy()
            x = x.cpu().detach().numpy()
            np.savez(f"Outputs/{experiment_name}/{module}/{f}/{i}.npz",x=x,y=y,attribution=attribution)
    if cfg['evaluate']:
        if not cfg["all_load"]:
            results_files = glob.glob(f"Outputs/{experiment_name}/{module}/{f}/*.npz")
            scores = {'del': [], 'ins': []}
            for file in results_files:
                results = np.load(file)
                attribution = results['attribution']
                x = torch.from_numpy(results['x']).to(device)
                deletion = CausalMetric(
                    model, 'del', substrate_fn=torch.zeros_like, hw=np.prod(INPUT_SIZE[-2:]), num_classes=NUM_CLASSES)
                insertion = CausalMetric(
                    model, 'ins', substrate_fn=torch.zeros_like, hw=np.prod(INPUT_SIZE[-2:]), num_classes=NUM_CLASSES)
                scores['del'].extend(deletion.evaluate(
                    x, attribution, len(x)).tolist())
                scores['ins'].extend(insertion.evaluate(
                    x, attribution, len(x)).tolist())
            scores['ins'] = np.array(scores['ins'])
            scores['del'] = np.array(scores['del'])
            with open(f'Outputs/{experiment_name}/{module}/{f}/scores.txt', 'w') as f:
                f.write("Insertion: " + str(scores['ins'].mean()) + "\n")
                f.write("Deletion: " + str(scores['del'].mean()) + "\n")
                print("Insertion: " + str(scores['ins'].mean()))
                print("Deletion: " + str(scores['del'].mean()))
        else:
            results_files = glob.glob(f"Outputs/{experiment_name}/{module}/{f}/*.npz")
            scores = {'del': [], 'ins': []}
            x_all = list()
            attribution_all = list()
            for file in results_files:
                results = np.load(file)
                attribution = results['attribution']
                x = torch.from_numpy(results['x']).to(device)
                x_all.append(x)
                attribution_all.append(attribution)
            x = torch.cat(x_all,dim=0)
            attribution = np.concatenate(attribution_all,axis=0)
            deletion = CausalMetric(
                model, 'del', substrate_fn=torch.zeros_like, hw=np.prod(INPUT_SIZE[-2:]), num_classes=NUM_CLASSES)
            insertion = CausalMetric(
                model, 'ins', substrate_fn=torch.zeros_like, hw=np.prod(INPUT_SIZE[-2:]), num_classes=NUM_CLASSES)
            scores['del'].extend(deletion.evaluate(
                x, attribution, 100).tolist())
            scores['ins'].extend(insertion.evaluate(
                x, attribution, 100).tolist())
            scores['ins'] = np.array(scores['ins'])
            scores['del'] = np.array(scores['del'])
            with open(f'Outputs/{experiment_name}/{module}/{f}/scores.txt', 'w') as f:
                f.write("Insertion: " + str(scores['ins'].mean()) + "\n")
                f.write("Deletion: " + str(scores['del'].mean()) + "\n")
                print("Insertion: " + str(scores['ins'].mean()))
                print("Deletion: " + str(scores['del'].mean()))
            
def pruning_runner(module,f,function,cfg):
    input_ = globals()[cfg["input"]]
    function_config = pruning_config.get(f,{})
    save_path = f"Outputs/{experiment_name}/{module}/{f}"
    if not os.path.exists(f"Outputs/{experiment_name}/{module}/{f}"):
        os.makedirs(f"Outputs/{experiment_name}/{module}/{f}")
    if not f == "greg":
        function = function(**function_config)
        ratio = function_config["pruning_ratio"]
        save_path = save_path + f"/{ratio}/mask.pkl"
        os.makedirs(f"Outputs/{experiment_name}/{module}/{f}/{ratio}",exist_ok=True)
        globals()[cfg["output"]] = function(model,train_dataloader,save_path)
    else:
        function = function(**function_config)
        ratio = function_config["stage_pr"]
        ratio = ratio[-1] if not model.is_single_branch else ratio.replace("[","").replace("]","")
        save_path = save_path + f"/{ratio}/mask.pkl"
        os.makedirs(save_path.replace("mask.pkl",""),exist_ok=True)
        globals()[cfg["output"]] = function(model,train_dataloader,test_dataloader,save_path)
        
        
def distribution(module):
    module = module.split("_")[0]
    return_dict = {
        "Mutants":mutants_runner,
        "Adversarial":adversarial_runner,
        "Pruning":pruning_runner,
        "Explanation":explanation_runner,
    }
    return return_dict[module]

if __name__ == "__main__":
    if TYPE == "Image Classification":
        last_module = "original"
        last_f = ""
        for module,func in pipeline_config.items():
            for f,cfg in func.items():
                if module.split('_')[0] != "Indicator":
                    md = importlib.import_module(module.split("_")[0])
                    function = getattr(md,f)
                    print(f"Running {f} from {module}")
                    runner = distribution(module)
                    runner(module,f,function,cfg)
                else:
                    md = importlib.import_module(f"{module.split('_')[0]}.{f}")
                    metric_function = md.metric
                    metrics, class_report = metric_function(globals()[cfg["model"]],globals()[cfg["input"]],num_classes=NUM_CLASSES)
                    if not os.path.exists(f"Outputs/{experiment_name}/Indicator/{last_module}_{last_f}"):
                        os.makedirs(f"Outputs/{experiment_name}/Indicator/{last_module}_{last_f}")
                    metrics.to_csv(f'Outputs/{experiment_name}/Indicator/{last_module}_{last_f}/metrics.csv', index=False)
                    class_report.to_csv(f'Outputs/{experiment_name}/Indicator/{last_module}_{last_f}/class_report.csv')
                last_f = f
            last_module = module
    elif TYPE == "Object Detection":
        last_module = "original"
        last_f = ""
        for module,func in pipeline_config.items():
            for f,cfg in func.items():
                if module.split('_')[0] != "Indicator":
                    md = importlib.import_module(module.split("_")[0])
                    function = getattr(md,f)
                    print(f"Running {f} from {module}")
                    runner = distribution(module)
                    runner(module,f,function,cfg)
                else:
                    md = importlib.import_module(f"{module.split('_')[0]}.{f}")
                    metric_function = md.ob_metric
                    output = metric_function(globals()[cfg["model"]],globals()[cfg["input"]])
                    if not os.path.exists(f"Outputs/{experiment_name}/Indicator/{last_module}_{last_f}"):
                        os.makedirs(f"Outputs/{experiment_name}/Indicator/{last_module}_{last_f}")
                    log_file = open(f'Outputs/{experiment_name}/Indicator/{last_module}_{last_f}/metrics.txt', 'w')
                    for line in output:
                        log_file.write(line)
                        log_file.write('\n')
                    log_file.close()
                last_f = f
            last_module = module
            
        