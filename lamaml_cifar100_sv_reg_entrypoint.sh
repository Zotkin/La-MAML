CIFAR='--data_path data/ --log_every 100 --dataset cifar100 --cuda --log_dir logs/'
SEED=$1

echo "SEED $1"

########## CIFAR DATASET Single-Pass ##########
##### La-MAML #####
python3 main.py $CIFAR --model lamaml_cifar --expt_name "lamaml_cifar_sv_reg_with_ratio_norm_memories_400_seed_$1" --memories 400 --batch_size 10 --replay_batch_size 10 --n_epochs 10 \
                    --opt_lr 0.25 --alpha_init 0.1 --opt_wt 0.1 --glances 1 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 5 --learn_lr --log_every 3125 --second_order --class_order random \
                    --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.1 --use-sv-regularization --ratio-factor 0.3 --norm-factor 0.05 \
