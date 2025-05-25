import os, torch, glob

def save_checkpoint(model, optim, epoch, out_dir, tag='best'):
    os.makedirs(out_dir, exist_ok=True)

    # имя файла: best.pth или final.pth или ckpt_###.pth
    if tag == 'best':
        fname = 'best.pth'              # перезаписываем один файл
    elif tag == 'final':
        fname = 'final.pth'
    else:
        fname = f'{tag}_{epoch:03d}.pth'

    torch.save({'model': model.state_dict(),
                'optim': optim.state_dict(),
                'epoch': epoch},
               os.path.join(out_dir, fname))

def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optim"])
    return model, optimizer, ckpt.get("epoch", 0) + 1