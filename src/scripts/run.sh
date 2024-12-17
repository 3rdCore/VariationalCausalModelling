
#########################################################
#This is where you can launch your code from the command line

# For more information on how to use Hydra, please refer to the documentation: https://hydra.cc/docs/intro
#########################################################

# For launching parallel jobs, you can use the following command with --multirun:
# This will start a naive grid search over the carthesian product of the arguments you provide

python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/unregistered_logs seed=0,1,2,3,4 my_custom_argument="config_1","config_2"

# For launching a single job, you can use the same command without --multirun. Make sure to provide a single value for each argument :

python train.py hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/unregistered_logs seed=0 my_custom_argument="config_1"


python train.py  --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/unregistered_logs seed=0,1,2 ++task.temperature=0.1,0.5,1,5 ++task.lr=1e-4,1e-3 ++task.is_beta_VAE=False ++SCM.interventional_shift=10 ++train_dataset.observational_density=0.1,0.5,0.8 ++train_dataset.shift=0.0,1.0 logger.tags=[hyp-search]
