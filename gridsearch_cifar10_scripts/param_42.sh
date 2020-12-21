docker run --gpus device=0 \
           --mount type=bind,source="$(pwd)"/data,target=/code/data \
           --mount type=bind,source="$(pwd)"/logs,target=/code/logs \
           lamaml bash lamaml_cifar100_sv_reg_entrypoint.sh 42