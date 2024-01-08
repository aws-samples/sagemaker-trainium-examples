# nemo-megatron-trn1-sagemaker

This repository helps in training Llama v2 models using Amazon SageMaker and Trn1 instances. The repo consists of 6 notebooks which has to be executed in order. 

1. Build custom Docker Image : This notebook builds a custom docker image which extends from Neuron Deep Learning Container for SageMaker and install NeuronX Nemo megatron on top of it.

2. Setup Fsx (optional) : This notebook helps in setting up FSx lustre which is recommended to use when running training on large cluster. The notebook creates a new VPC / subnet and fsx lustre within the subnet. The notebook also adds an S3 bucket as a Data Repository Association (DRA) to the Fsx Lustre file system which enables 2 way sync between FSx and S3 for Create/delete/update file operations.

3. Prepare Dataset : The notebook takes an Huggingface dataset name , downloads from the hub and tokenizes the data. The notebook converts the tokenized data to Nemo Format which will be used for training. 

4. Convert HF weights to Nemo Checkpoint (optional) : This notebook helps in converting the pretrained weights from HuggingFace hub into Nemo checkpoints. This is useful when you are fine tuning a model and want to start with pretrained weights. The notebook uses llama2 7b as example. 

5. Train Llama V2 : The notebook contains parameters and scripts to train llama2 using Trn1 instances. 

6. Convert Nemo Checkpoints to HF : This Notebook helps to convert the checkpoints saved by Nemo into .pt file which can be used to deploy for downstream inference. 