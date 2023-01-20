neuron_parallel_compile XLA_DOWNCAST_BF16=1 torchrun --nproc_per_node=32 \
dp_bert_large_hf_pretrain_hdf5.py \
--data_dir=$SM_CHANNEL_TRAINING \
--steps_this_run 10 \
--batch_size 16 \
--grad_accum_usteps 32