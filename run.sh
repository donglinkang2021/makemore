python makemore.py --multirun \
    model=transformer,bigram,bow,mlp,rnn,gru \
    system.max_steps=20000 \
    system.input_file=names.txt \
    system.work_dir=names
python makemore.py --multirun \
    model=transformer,bigram,bow,mlp,rnn,gru \
    system.input_file=names.txt \
    system.work_dir=names \
    system.sample_only=true
# nohup bash run.sh > run.log 2>&1 &
# nohup tensorboard --logdir=names > tensorboard.log 2>&1 &