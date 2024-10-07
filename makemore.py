import os
import time
import hydra
from config import Config
from data import create_datasets, InfiniteDataLoader
from utils import evaluate, _print_samples
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def get_model_name(cfg:Config):
    model_name = cfg.model._target_.split('.')[-1]
    if model_name == 'RNN':
        model_name += f"_{cfg.model.cell_type}"
    return model_name

@hydra.main(config_path='configs', config_name='default', version_base=None)
def main(cfg:Config):
    # from omegaconf import DictConfig, OmegaConf
    # print(OmegaConf.to_yaml(cfg))
    # return
    model_name = get_model_name(cfg)

    # system inits
    torch.manual_seed(cfg.system.seed)
    torch.cuda.manual_seed_all(cfg.system.seed)
    import datetime
    from pathlib import Path
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(cfg.system.work_dir) / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # init datasets
    train_dataset, test_dataset = create_datasets(cfg.system.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"dataset determined that: {vocab_size=}, {block_size=}")

    # init model
    if model_name == 'Bigram':
        model:nn.Module = hydra.utils.instantiate(cfg.model, vocab_size=vocab_size)
    else:
        model:nn.Module = hydra.utils.instantiate(cfg.model, block_size=block_size, vocab_size=vocab_size)
    model.to(cfg.system.device)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    if cfg.system.resume or cfg.system.sample_only: # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(log_dir, f'{model_name}.pt')))

    def print_samples(num:int=10):
        _print_samples(
            model=model, train_dataset=train_dataset,
            test_dataset=test_dataset,
            num=num, top_k=cfg.sampling.top_k,
            device=cfg.system.device
        )

    if cfg.system.sample_only:
        print_samples(50)
        return

    # init optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.optimization.learning_rate, 
        weight_decay=cfg.optimization.weight_decay, 
        betas=(0.9, 0.99), 
        eps=1e-8
    )

    # init dataloader
    batch_loader = InfiniteDataLoader(
        train_dataset, 
        batch_size=cfg.optimization.batch_size, 
        pin_memory=True, 
        num_workers=cfg.system.num_workers
    )

    # training loop
    best_loss = None
    step = 0
    while True:

        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(cfg.system.device) for t in batch]
        X, Y = batch

        # feed into the model
        logits, loss = model(X, Y)

        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if cfg.system.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        # evaluate the model
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10)
            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(log_dir, f'{model_name}.pt')
                print(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        # sample from the model
        if step > 0 and step % 200 == 0:
            print_samples(num=10)

        step += 1
        # termination conditions
        if cfg.system.max_steps >= 0 and step >= cfg.system.max_steps:
            break

if __name__ == '__main__':
    main()