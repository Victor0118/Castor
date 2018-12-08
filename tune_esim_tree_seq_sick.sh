for regularization in 1e-4 5e-4 5e-5 1e-6 1e-5
do
  for lr_reduce_factor in 0.3 0.1 0.5
  do
    for dropout in 0.5 0.1 0.3 0.2 0.4 0
    do
        for lr in 0.0004 0.0002 0.001 0.00005
        do
            logfile="log_sick_tree/esim_tree_seq_sick_${regularization}_${lr}_${lr_reduce_factor}_${dropout}.log"
            echo "Log at: ${logfile}"
            python -m esim esim_tree_seq.sick.model_tune --dataset sick --arch seq-tree --epochs 25 --regularization ${regularization} --lr ${lr} --batch-size 64 --lr-reduce-factor ${lr_reduce_factor} --dropout ${dropout} &> ${logfile};
        done
    done
  done
done
