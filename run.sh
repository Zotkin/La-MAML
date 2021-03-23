 FIRST_INCREMENT=5;
 INCREMENT=5
 MEMORIES=(50 100 200 400) ;
 for m in ${MEMORIES[@]};
  do sbatch run_slurm_v2.sh "$FIRST_INCREMENT" "$m" "$INCREMENT";
  done