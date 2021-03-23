#!/bin/bash
#SBATCH -n 4                  # number of tasks
#SBATCH -N 1                  # number of nodes
#SBATCH -n_tasks_per_node 4
#SBATCH -c 2                  # number of cores per task
#SBATCH -t 7-0                # time limit ; format : "minutes:seconds" | "hours:minutes:seconds" | "days-hours"
#SBATCH -p GPU                # partition to use
#SBATCH --gres=gpu:1          # total number of gpu to allocate
#SBATCH --mem=20G             # maximum amout of RAM that can be used
#SBATCH -x calcul-gpu-lahc-3


CIFAR='--data_path data/ --log_every 100 --dataset cifar100 --cuda --log_dir logs/'
SEED=0
MEMORIES=(50 100 200 400)
FIRST_INCREMENT=50

export OMP_NUM_THREADS=4
source /home/zoy07590/virtenv_project-lamaml/bin/activate

for ((i = 0 ; i < ${#MEMORIES[@]}; i++)); do
  srun python main.py $CIFAR --model lamaml_cifar \
                      --expt_name lamaml_cifar_baseline_"$FIRST_INCREMENT"_increment_5_memories_"${MEMORIES[i]}"\
                      --memories "${MEMORIES[i]}" \
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
                      --first-increment $FIRST_INCREMENT &
done
wait



