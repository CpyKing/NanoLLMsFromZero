import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nanoGPT import GPT, GPTConfig
from nanoGPT_dataset import MyDataset

def train(model, optimizer, scheduler, train_loader, device):
    model.train()
    total_loss = 0
    for iters, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, targets=y)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        if iters % 100 == 0:
            print(f"iter {iters}, loss {loss}")
    return total_loss

def eval(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            val_loss += loss.item()
    return val_loss

if __name__ == '__main__':
    config = GPTConfig()
    model = GPT(GPTConfig)
    device = 'cuda'
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    # print(total_params / 1e6, 'M')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    dataset = MyDataset(path='./data/chinese-poem.json')
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    
    
    
    for epoch in range(10):
        train_loss = train(model, optimizer, scheduler, train_loader, "cuda")
        val_loss = eval(model, val_loader, "cuda")
        
        print(f'Epoch {epoch}, train loss {train_loss / len(train_loader)}, val loss {val_loss / len(val_loader)}')
        avg_val_loss = val_loss / len(val_loader)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
        }
        # 保存每个epoch的模型
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')
    
    model.load_state_dict(torch.load('./checkpoints/model_epoch_3.pt')['model_state_dict'])
    model.eval()
    eval_text = '这是一个非常搞笑的笑话，'
    eval_text_enc = dataset.encode(eval_text)
    eval_text_enc = torch.tensor(eval_text_enc, device=device)
    eval_text_enc = eval_text_enc.unsqueeze(dim=0)
    res = model.generate(eval_text_enc, max_new_tokens=512)
    print(dataset.decode(res[0,:].tolist()))