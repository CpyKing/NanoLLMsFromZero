import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import pyarrow.parquet as pq

from llama3.model import Transformer, ModelArgs
# from nanollama3 import Transformer, ModelArgs
from llama3.tokenizer import Tokenizer

from typing import Sequence
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)

class MyDataset(Dataset):
    def __init__(self, tokenizer_path: str, max_seq_len: int):
        super().__init__()
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        self.enc_data = read_dataset('/home/zzz/dataset/train-00016-of-00383.parquet', self.tokenizer, max_seq_len)
    
    def __len__(self):
        return len(self.enc_data)

    def __getitem__(self, idx):
        chunk = self.enc_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    
    def encode(self, text, bos: bool=False, eos: bool=False):
        return self.tokenizer.encode(text, bos=bos, eos=eos)
    
    def decode(self, ids: Sequence[int]):
        return self.tokenizer.decode(ids)
        

def read_dataset(path: str, tokenizer: Tokenizer, max_seq_len: int):
    parquet_file = pq.ParquetFile(path)

    table = parquet_file.read()
    df = table.to_pandas()
    
    raw_text = []
    for idx, row in df.iterrows():
        if idx > 60000:
            break
        if idx > 50000:
            raw_text.append(row['text'])
    raw_text_enc = []
    for row in raw_text:
        raw_text_enc.extend(tokenizer.encode(row, bos=True, eos=True))
    
    enc_data = []
    for i in range(0, len(raw_text_enc), max_seq_len):
        enc_item = raw_text_enc[i: i + max_seq_len + 1]
        if len(enc_item) < max_seq_len + 1:
            enc_item = enc_item + [tokenizer.eos_id] * (max_seq_len + 1 - len(enc_item))
        enc_data.append(enc_item)
    return enc_data
    
    

def main():
    model_args: ModelArgs = ModelArgs(
        dim=1024,   # 4096
        n_layers=8,    # 32
        n_heads=8,     # 32
        n_kv_heads=4,   # 8
        vocab_size=128256,  # 128256
        multiple_of=256,   # 1024
        ffn_dim_multiplier=1.3, #1.3
        norm_eps=1e-05,         # 1e-05
        rope_theta=500000.0,    # 500000.0
        
        max_seq_len=512,    # 512
        max_batch_size=10,  # 10
    )
    device = 'cuda:0'
    model = Transformer(model_args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params / 1e6, 'M')
    dataset = MyDataset('/home/zzz/model/Meta-Llama-3-8B-Instruct/original/tokenizer.model', model_args.max_seq_len)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=model_args.max_batch_size, shuffle=True)
    eval_loader = DataLoader(val_dataset, batch_size=model_args.max_batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    print(f'len dataset {len(train_loader)}')
    
    model.load_state_dict(torch.load('./checkpoints/model_epoch_5.pt')['model_state_dict'])
    # optimizer.load_state_dict(torch.load('./checkpoints/model_epoch_1.pt')['optimizer_state_dict'])
    # scheduler.load_state_dict(torch.load('./checkpoints/model_epoch_1.pt')['scheduler_state_dict'])
    # for epoch in range(10):
    #     print(f'Epoch {epoch} begin ...')
    #     train_loss = train(model, train_loader, model_args, optimizer, scheduler, device, dataset.tokenizer.eos_id)
    #     val_loss = eval(model, eval_loader, device, dataset.tokenizer.eos_id)
    #     print(f'Epoch {epoch} end, train loss {train_loss}, val loss {val_loss}')
    #     checkpoint = {
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'scheduler_state_dict': scheduler.state_dict(),
    #         'val_loss': val_loss,
    #     }
    #     torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')
    
    
    model.eval()
    eval_text = 'To copy construct from a tensor, it is recommended'
    eval_text_enc = dataset.encode(eval_text)
    eval_text_enc = torch.tensor(eval_text_enc, device=device)
    eval_text_enc = eval_text_enc.unsqueeze(dim=0)
    res = model.generate(eval_text_enc, model_args.max_seq_len, 64, dataset.tokenizer.eos_id, dataset.tokenizer.eos_id, device)
    print(dataset.decode(res[0,:].tolist()))
    
def eval(model, val_loader, device='cuda:0', ignore_index=-100):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            loss = model.forward_train(x, y, ignore_index)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def train(model: Transformer, train_loader: DataLoader, model_args: ModelArgs, optimizer, scheduler, device='cuda:0', ignore_index=-100):
    model.train()
    total_loss = 0
    time_begin = time.time()
    for iter, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        loss = model.forward_train(x, y, ignore_index)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if iter % 50 == 0:
            print(f'Iter {iter}, loss {loss}, duration {(time.time()-time_begin)/50}s')
            time_begin = time.time()
    return total_loss / len(train_loader)
        
        
    

if __name__ == '__main__':
    # if not torch.distributed.is_initialized():
    #     torch.distributed.init_process_group("nccl")
    # model_parallel_size = None
    # if not model_parallel_is_initialized():
    #     if model_parallel_size is None:
    #         model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
    #     initialize_model_parallel(model_parallel_size)
    
    main()