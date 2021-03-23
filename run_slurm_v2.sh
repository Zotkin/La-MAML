#!/bin/bash
#SBATCH -n 1                  # number of tasks
#SBATCH -N 1                  # number of nodes
#SBATCH -c 1                  # number of cores per task
#SBATCH -t 7-0                # time limit ; format : "minutes:seconds" | "hours:minutes:seconds" | "days-hours"
#SBATCH -p GPU                # partition to use
#SBATCH --gres=gpu:1          # total number of gpu to allocate
#SBATCH --mem=20G             # maximum amout of RAM that can be used
#SBATCH -x calcul-gpu-lahc-3


CIFAR='--data_path data/ --log_every 100 --dataset cifar100 --cuda --log_dir logs/'
SEED=0

FIRST_INCREMENT="$1"
MEMORY="$2"
INCREMENT="$3"

export OMP_NUM_THREADS=1
source /home/zoy07590/virtenv_project-lamaml/bin/activate
srun python main.py $CIFAR --model lamaml_cifar \
                      --expt_name lamaml_cifar_baseline_"$FIRST_INCREMENT"_increment_"$INCREMENT"_memories_"$MEMORY"\
                      --memories "$MEMORY" \
                      --batch_size 10 \
                      --replay_batch_size 10 \
                      --n_epochs 10 \
                      --opt_lr 0.25 \
                      --alpha_init 0.1 \
                      --opt_wt 0.1 \
                      --glances 1 \
                      --loader class_incremental_loader \
                      --increment $INCREMENT \
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
                      --first-increment $FIRST_INCREMENT



