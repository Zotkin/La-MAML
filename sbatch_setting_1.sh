#!/bin/bash
#SBATCH -n 4                  # number of tasks
#SBATCH -N 1                  # number of nodes
#SBATCH -c 4                  # number of cores per task
#SBATCH -t 7-0                # time limit ; format : "minutes:seconds" | "hours:minutes:seconds" | "days-hours"
#SBATCH -p GPU                # partition to use
#SBATCH --gres=gpu:1          # total number of gpu to allocate
#SBATCH --mem=10G             # maximum amout of RAM that can be used
#SBATCH -x calcul-gpu-lahc-2

CIFAR='--data_path data/ --log_every 100 --dataset cifar100 --cuda --log_dir logs/'
SEED=0
MEMORIES=400
FIRST_INCREMENT=50

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source /home_expes/tools/python/Python-3.8.7_gpu/bin/activate

srun python main.py $CIFAR --model lamaml_cifar \
                      -expt_name lamaml_cifar_baseline_"$FIRST_INCREMENT"_memories_"$MEMORIES"\
                      --memories $MEMORIES \
                      --batch_size 10 \
                      --replay_batch_size 10 \
                      --n_epochs 10 \
                      --opt_lr 0.25 \
                      --alpha_init 0.1 \
                      --opt_wt 0.1 \
                      --glances 1 \
                      --loader class_incremental_loader \
                      --increment 5 \
                      --arch "pc_cnn" \
                      --cifar_batches 5 \
                      --learn_lr \
                      --log_every 3125 \
                      --second_order \
                      --class_order random \
                      --seed $SEED \
                      --grad_clip_norm 1.0 \
                      --calc_test_accuracy \
                      --validation 0.1 \
                      --first-incremet $FIRST_INCREMENT
