# fast_imagenet_ffcv
Fast ImageNet training code with FFCV

## Setting up
Installation: install pytorch and ffcv (https://github.com/libffcv/ffcv).

Run via `python train_ffcv.py`.

## Notes
The datasets are already preprocessed (available in `/self/scr-sync/nlp/imagenet_ffcv` on each NLP cluster machine).
Training is fast bc entire ImageNet is cached in memory.
Tested on 4 3090s, training is 10 epochs/hour (10 hours to train a model on 4 gpus).
Based on inital estimates, it seems like it would take 1.5 daays (~36 hours) on 1 3090 gpu.

## Slurm advice
Since dataset itself is ~70G, I would request 100G for the job to be safe.
I would also request `# of cpus = 2 * (num_workers + 1) * num_gpus`.
This is since each gpu requires `num_workers` dataloading threads and one thread for the model. 
Multiply by 2 since the dataloading transforms are JITed so hyperthreading doesn't give much perf.
This means for the machines with only 32 slurm cpus, probably best to use 3 workers per gpu (but you should try for yourself).
Otherwise, I found 4 workers per gpu with total # cpus request 40 to give fast speeds on the 3090 setup.

The slurm command I used for the 3090s was `nlprun -r 100G -c 40 -g 4`.
