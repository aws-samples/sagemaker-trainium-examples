# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

         

import argparse
import os
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
# Initialize XLA process group for torchrun
import torch_xla.distributed.xla_backend
import random
import evaluate



device = "xla"
torch.distributed.init_process_group(device)
world_size = xm.xrt_world_size() 


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    parser.add_argument(
        "--train_file", type=str, default=os.path.join(os.environ["SM_CHANNEL_TRAIN"],"train.csv"), help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=os.path.join(os.environ["SM_CHANNEL_VAL"],"test.csv"), help="A csv or a json file containing the     validation data."
    )
    

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
  
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,   
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
   
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )


    parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"], help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=100, help="A seed for reproducible training.")
    args = parser.parse_args()
    

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need a training/validation file.")

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])

    print("Local rank {} , World Rank {} , World Size {}".format(args.local_rank,args.world_rank,args.world_size))

    return args

def gather(tensor, name="gather tensor"):
    return xm.mesh_reduce(name, tensor, torch.cat)

def main():


    args = parse_args()

    datafiles = {}
    datafiles["train"] = args.train_file
    datafiles["eval"] = args.validation_file 

    dataset = load_dataset("csv",data_files=datafiles)
   

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch['content'], max_length=args.max_length, padding='max_length', truncation=True)
    
    # load dataset
    train_dataset = dataset['train'].shuffle()
    eval_dataset = dataset['eval'].shuffle()

    # tokenize dataset only on one process to avoid repetition
    if args.world_rank == 0:
        train_dataset = train_dataset.map(tokenize, batched=True)
        eval_dataset = eval_dataset.map(tokenize, batched=True)
        train_dataset.save_to_disk("/tmp/train_ds/")
        eval_dataset.save_to_disk("/tmp/eval_ds/")

    xm.rendezvous("wait_for_everyone_to_reach")

    if args.world_rank != 0:
        train_dataset = load_from_disk("/tmp/train_ds/")
        eval_dataset = load_from_disk("/tmp/eval_ds/")
    
    xm.rendezvous("wait_for_everyone_to_reach")

    # Log a few random samples from the training set:

    
    # set format for pytorch
    train_dataset =  train_dataset.rename_column("label", "labels")
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    if args.world_rank == 0:
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]}.")

    eval_dataset =  eval_dataset.rename_column("label", "labels")
    eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    if args.world_rank == 0:
        for index in random.sample(range(len(eval_dataset)), 3):
            print(f"Sample {index} of the training set: {eval_dataset[index]}.")

    
    # Set up distributed data loader
    train_sampler = None
    if world_size > 1: # if more than one core
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas = args.world_size,
            rank = args.world_rank,
            shuffle = True,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size = args.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler else True,
    )

    if world_size > 1: # if more than one core
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas = args.world_size,
            rank = args.world_rank,
            shuffle = True,
        )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size = args.per_device_eval_batch_size,
        sampler=eval_sampler,
        shuffle=False if eval_sampler else True,
    )


    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    eval_device_loader = pl.MpDeviceLoader(eval_loader, device)
    num_training_steps = args.num_train_epochs * len(train_device_loader)
    progress_bar = tqdm(range(num_training_steps))

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Get the metric function
    metric = evaluate.load("accuracy")
    
    for epoch in range(args.num_train_epochs):
        model.train() 
        for batch in train_device_loader:
            batch = {k: v.to(device) for k, v, in batch.items()}
            outputs = model(**batch)
            optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            xm.optimizer_step(optimizer) #gather gradient updates from all cores and apply them
            if args.world_rank == 0:
                progress_bar.update(1)
        if args.world_rank == 0:
            print(
            "Epoch {}, rank {}, Loss {:0.4f}".format(epoch, args.world_rank, loss.detach().to("cpu"))
            )
        # Run evaluation after each epochs
        model.eval()
        if args.world_rank == 0:
            print("Running evaluation for the model")
        for eval_batch in enumerate(eval_device_loader):
            with torch.no_grad():
                batch = {k: v.to(device) for k, v, in batch.items()}
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            xm.rendezvous("wait_for_everyone_to_reach")
            # Gather predictions and labels from all workers to compute accuracy.
            predictions = gather(predictions)
            references = gather(batch["labels"])
            metric.add_batch(
                predictions=predictions,
                references=references
            ) 
        eval_metric = metric.compute()
        if args.world_rank == 0:
            print(f"Validation Accuracy - epoch {epoch}: {eval_metric}")
            

    # Save checkpoint for evaluation (xm.save ensures only one process save)
    if args.output_dir is not None:
        xm.save(model.state_dict(), f"{args.output_dir}/checkpoint.pt")
        if args.world_rank == 0:
            tokenizer.save_pretrained(args.output_dir)
    if args.world_rank == 0:
        print('----------End Training ---------------')
    
if __name__ == '__main__':
    main()