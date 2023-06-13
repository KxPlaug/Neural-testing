import toml

adversarial_config = toml.load("Config/adversarial.toml")
explanation_config = toml.load("Config/explanation.toml")
mutants_config = toml.load("Config/mutants.toml")
pruing_config = toml.load("Config/pruning.toml")
pipeline_config = toml.load("Config/pipeline.toml")