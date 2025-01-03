
#########################################################
#This is where you can launch your code from the command line

# For more information on how to use Hydra, please refer to the documentation: https://hydra.cc/docs/intro
#########################################################

# For launching parallel jobs, you can use the following command with --multirun

# For launching a single job, you can use the same command without --multirun. Make sure to provide a single value for each argument.

python train.py  --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/unregistered_logs seed=0,1,2,3,4,5,6,7,8,9 ++task.temperature=0.5 ++task.is_beta_VAE=False ++SCM.interventional_shift=10 ++train_dataset.shift=1.0 logger.tags=[experiment]
