import os

base_path = "pipelines"

dirs = ["resnet50_pruning_finished_taylor","vgg16_pruning_finished_taylor"]

# result_sh = "run_obd_after_pruning.sh"
result_sh = "run_taylor_after_pruning.sh"
file = open(result_sh,"w")

for dir in dirs:
    toml_files = os.listdir(os.path.join(base_path,dir))
    sorted(toml_files)
    toml_files = reversed(toml_files)
    for toml_file in toml_files:
        if toml_file.endswith(".toml"):
            print(os.path.join(base_path,dir,toml_file))
            # os.system("python3 run_after_pruning.py --config_file " + os.path.join(base_path,dir,toml_file))
            toml_file_path = os.path.join(base_path,dir,toml_file)
            # print(toml_file_path)
            print("python main.py --pipeline " + toml_file_path)
            file.write("python main.py --pipeline " + toml_file_path + "\n")
            