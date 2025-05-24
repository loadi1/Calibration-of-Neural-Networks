import torch, os

def save_checkpoint(model, optim, epoch, out_dir, tag='ckpt'):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({'model': model.state_dict(), 'optim': optim.state_dict(), 'epoch': epoch},
               os.path.join(out_dir, f'{tag}_{epoch:03d}.pth'))

def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optim"])
    return model, optimizer, ckpt.get("epoch", 0) + 1