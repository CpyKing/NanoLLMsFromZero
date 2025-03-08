import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tiktoken
import json

class MyDataset(Dataset):
    def __init__(self, path='./data/chinese-poem.json', block_size=512):
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size # max len
        
        self.encoded_data = []
        # special token to split text
        # <|endoftext|>
        self.eos_token = self.enc.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]
        
        raw_data = []
        max_line = 500000000
        with open(path, 'r') as f:
            # file_data = json.load(f)
            for i, line in enumerate(f):
                if i > max_line:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    # text = line['text'].strip()
                    raw_data.append(text)
                except Exception as e:
                    continue
            
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])  
        
        # block size split, max len is 512
        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i: i + self.block_size + 1]    # 512 每一行其实是 513
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)
        
    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    
    def encode(self, text):
        return self.enc.encode(text)
    
    def decode(self, ids):
        return self.enc.decode(ids)
            