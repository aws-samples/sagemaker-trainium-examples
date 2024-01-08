from omegaconf.omegaconf import OmegaConf
from hydra import compose, initialize
from megatron_gpt_pretraining import main
import argparse
import os


def start() -> None:
    parser = argparse.ArgumentParser()


    _,all = parser.parse_known_args()
    overrides = []
    for i in range(0,len(all),2):
        overrides.append(all[i].replace("--","")+"="+str(all[i+1]))
    
    # global initialization
    initialize(version_base=None, config_path="conf", job_name="nemo_neuron_app")
    cfg = compose(config_name="megatron_llama_config", overrides=overrides)

    if int(os.environ["RANK"]) == 0:
        print("*************Logging configuration for Training**************")
        print(OmegaConf.to_yaml(cfg))

    main(cfg)


if __name__ == '__main__':
    start()
