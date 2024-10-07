python makemore.py --multirun \
    model=transformer,bigram,bow,mlp,rnn,gru \
    system.max_steps=20000 \
    system.input_file=data/names_en.txt \
    system.work_dir=names/en > logs/train.en.log
python makemore.py --multirun \
    model=transformer,bigram,bow,mlp,rnn,gru \
    system.input_file=data/names_en.txt \
    system.work_dir=names/en \
    system.sample_only=true > logs/eval.en.log
python makemore.py --multirun \
    model=transformer,bigram,bow,mlp,rnn,gru \
    system.max_steps=20000 \
    system.input_file=data/names_zh.txt \
    system.work_dir=names/zh > logs/train.zh.log
python makemore.py --multirun \
    model=transformer,bigram,bow,mlp,rnn,gru \
    system.input_file=data/names_zh.txt \
    system.work_dir=names/zh \
    system.sample_only=true > logs/eval.zh.log
python makemore.py --multirun \
    model=transformer,bigram,bow,mlp,rnn,gru \
    system.max_steps=20000 \
    system.input_file=data/names_ja.txt \
    system.work_dir=names/ja > logs/train.ja.log
python makemore.py --multirun \
    model=transformer,bigram,bow,mlp,rnn,gru \
    system.input_file=data/names_ja.txt \
    system.work_dir=names/ja \
    system.sample_only=true > logs/eval.ja.log
# nohup bash run.sh > run.log 2>&1 &
# nohup tensorboard --logdir=names > tensorboard.log 2>&1 &