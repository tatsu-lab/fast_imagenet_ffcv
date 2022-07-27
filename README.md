# fast_imagenet_ffcv
Fast ImageNet training code with FFCV

## Setting up
Installation: install pytorch and ffcv (https://github.com/libffcv/ffcv).

Run via `python train_ffcv.py`.

## Notes
The datasets are already preprocessed (available in `/self/scr-sync/nlp/imagenet_ffcv` on each NLP cluster machine).
Training is fast bc entire ImageNet is cached in memory.
Tested on 4 3090s, it takes 10 hours to train (~10 epochs/hour).
On 1 3090, it takes 36 hours.
On 4 A100s, it takes 5 hours.

## Slurm advice
Since dataset itself is ~70G, I would request 100G for the job to be safe.
I would also request `# of cpus = 2 * (num_workers + 1) * num_gpus`, so 40 cpus for 4 gpus.
This is since each gpu requires `num_workers` dataloading threads and one thread for the model. 
Multiply by 2 since the dataloading transforms are JITed so hyperthreading doesn't give much perf.
This means for the machines with only 32 slurm cpus, probably best to use 3 workers per gpu (but you should try for yourself).
Otherwise, I found 4 workers per gpu with total # cpus request 40 to give fast speeds on the 3090 setup.

## How I wrote this
This is essentially just the standard PyTorch ImageNet training script (https://github.com/pytorch/examples/blob/main/imagenet/main.py) with a modified dataloader. The meat of the dataloader bits are here: https://github.com/tatsu-lab/fast_imagenet_ffcv/blob/main/train_ffcv.py#L123-L154.
