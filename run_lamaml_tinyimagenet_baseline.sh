docker run --gpus device=1 \
           --mount type=bind,source="$(pwd)"/data,target=/code/data \
           --mount type=bind,source="$(pwd)"/logs,target=/code/logs \
           lamaml bash lamaml_tinyimagenet_entrypoint.sh
