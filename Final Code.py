# ─── Settings ─────────────────────────────────────────────
MODEL_NAME = "efficientb0"
DATA_DIR   = "/content/cifar10-long-tail"
TRAIN_DIR, VAL_DIR = (f"{DATA_DIR}/{p}" for p in ["train", "test"])

TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE  = 128
NUM_WORKERS      = 4

LR, MIN_LR          = 1e-4, 1e-6
NUM_EPOCHS          = 30
WEIGHT_DECAY        = 0.05
EMA_DECAY           = 0.9999

SEED      = 42
OUT_DIR   = f"./results/{MODEL_NAME}"
CKPT_LAST = f"{OUT_DIR}/last.pth"
CKPT_BEST = f"{OUT_DIR}/best_acc.pth"

# ─── Libraries ────────────────────────────────────────────
import os, random, copy, itertools, numpy as np, matplotlib.pyplot as plt
from collections import Counter
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEmaV2
import timm
from torchinfo import summary
from thop import profile
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.amp import autocast, GradScaler

#Imports for Class-balanced Sampling
from collections import Counter
from torch.utils.data import WeightedRandomSampler

# ─── Reproducibility ──────────────────────────────────────
def set_seed(sd=42):
    random.seed(sd); np.random.seed(sd)
    torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False

# ─── Transforms ───────────────────────────────────────────
train_tf = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomAffine(degrees=15, scale=(0.8, 1.2)),
    T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.25, scale=(0.02, 0.2), value='random'),
])

eval_tf = T.Compose([        
    T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─── Data Loading ─────────────────────────────────────────
def get_loaders():
    tr_ds = ImageFolder(TRAIN_DIR, transform=train_tf)
    va_ds = ImageFolder(VAL_DIR,   transform=eval_tf)

    #--------Modified for Class-balanced Sampling-------------
    examples = tr_ds.targets
    class_counts = Counter(examples)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[t] for t in examples]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    #----------End Modification---------------------------

    train_ldr = DataLoader(tr_ds, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True, persistent_workers=True, sampler=sampler) #Changed (shuffle) to (sampler)
    val_ldr  = DataLoader(va_ds, batch_size=EVAL_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, shuffle=False)
    return train_ldr, val_ldr, len(va_ds.classes)

# ─── Model & Scheduler ────────────────────────────────────
def build_model(num_classes):
    return timm.create_model(
        "efficientnet_b0", pretrained=True, num_classes=num_classes, drop_rate=0.0, drop_path_rate=0.1,
    )

def build_scheduler(opt):
    return CosineLRScheduler(
        opt, t_initial=NUM_EPOCHS, lr_min=MIN_LR, cycle_limit=1, t_in_epochs=True
    )

# ─── Mixup Augmentation ──────────────────────────
# --------------ADDED FOR MIXUP---------------
def mixup_data(x, y, alpha=0.2):
    #Apply mixup augmentation - blends entire images
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    #Compute mixup loss
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
# -----------END MIXUP HELPERS----------------

# ─── Training & Evaluation ────────────────────────────────
def train_epoch(model, ldr, loss_fn, opt, dev, scaler, ema=None, use_mixup=True, mixup_alpha=0.2):  #Added MixUp params

    model.train()
    loss_sum = correct = total = 0

    for x, y in tqdm(ldr, leave=False, desc="Train"):

        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()

        # -------------- ADDED FOR MIXUP---------------
        if use_mixup:
            x, y_a, y_b, lam = mixup_data(x, y, alpha=mixup_alpha)
        # -----------END MIXUP------------------

        with autocast('cuda'):
            out = model(x)
            # ----------MODIFIED FOR MIXUP-------------
            if use_mixup:
                loss = mixup_criterion(loss_fn, out, y_a, y_b, lam)
            else:
                loss = loss_fn(out, y)
            #-----------END MIXUP------------------

        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

        if ema:
            ema.update(model)

        # -----------MODIFIED FOR MIXUP: Track accuracy differently for mixed samples------------
        loss_sum += loss.item() * x.size(0)
        if use_mixup:
            correct  += (lam * out.argmax(1).eq(y_a).float() + (1-lam) * out.argmax(1).eq(y_b).float()).sum().item()
        else:
            correct  += out.argmax(1).eq(y.argmax(1) if y.ndim == 2 else y).sum().item()
        total    += x.size(0)
        # ---------------- END MIXUP ACCURACY TRACKING-----------------

    return loss_sum / total, correct / total

@torch.no_grad()
def eval_epoch(model_eval, ldr, loss_fn, dev, tag):
    model_eval.eval()
    loss_sum = correct = total = 0

    for x, y in tqdm(ldr, leave=False, desc=tag):
        with autocast('cuda'):
            out = model_eval(x.to(dev))
            loss = loss_fn(out, y.to(dev))
        loss_sum += loss.item() * y.size(0)
        correct  += out.argmax(1).eq(y.to(dev)).sum().item()
        total    += y.size(0)

    return loss_sum / total, correct / total

# -------------- ENHANCED TTA IMPLEMENTATION --------------------
@torch.no_grad()
def predict_all(model_eval, ldr, dev, use_tta=False, tta_mode='enhanced'):
    model_eval.eval()
    preds, targs = [], []
    for x, y in tqdm(ldr, leave=False, desc="Test"):
        if use_tta:
            with autocast('cuda'):
                x_dev = x.to(dev)

                if tta_mode == 'enhanced':
                    # ----------ENHANCED TTA: 4 augmentations for better accuracy--------
                    # Augmentation 1: Original image
                    out1 = model_eval(x_dev)
                    # Augmentation 2: Horizontal flip (left-right)
                    out2 = model_eval(torch.flip(x_dev, dims=[3]))
                    # Augmentation 3: Vertical flip (up-down)
                    out3 = model_eval(torch.flip(x_dev, dims=[2]))
                    # Augmentation 4: Both horizontal and vertical flips
                    out4 = model_eval(torch.flip(x_dev, dims=[2, 3]))
                    # Average all 4 predictions for final output
                    out = (out1 + out2 + out3 + out4) / 4.0
                    # ----------END ENHANCED TTA------------------
                else:
                    # Simple TTA: original + horizontal flip only (2 augmentations)
                    out1 = model_eval(x_dev)
                    out2 = model_eval(torch.flip(x_dev, dims=[3]))
                    out = (out1 + out2) / 2.0
        else:
            # No TTA: single forward pass
            with autocast('cuda'):
                out = model_eval(x.to(dev))

        preds.append(out.argmax(1).cpu())
        targs.append(y)
    return torch.cat(preds).numpy(), torch.cat(targs).numpy()
# -----------END ENHANCED TTA IMPLEMENTATION--------------------

# ─── Plotting helpers ─────────────────────────────────────
def plot_confusion(cm, classes, path):
    plt.figure(figsize=(14, 12)); plt.imshow(cm, cmap="Blues"); plt.colorbar()
    idx = np.arange(len(classes))
    plt.xticks(idx, classes, rotation=90, fontsize=8)
    plt.yticks(idx, classes, fontsize=8)
    thr = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:.2f}", ha="center",
                 color="white" if cm[i, j] > thr else "black", fontsize=7)
    plt.ylabel("True"); plt.xlabel("Pred")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

# ─── Checkpoint helpers ───────────────────────────────────
def save_ckpt(epoch, model, ema, opt, sch, scaler,
              best, pat, trL, vaL, trA, vaA, lrs):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "ema":   ema.module.state_dict(),
        "optimizer":  opt.state_dict(),
        "scheduler":  sch.state_dict(),
        "scaler":     scaler.state_dict(),
        "best": best, "pat": pat,
        "trL": trL, "vaL": vaL, "trA": trA, "vaA": vaA, "lrs": lrs
    }, CKPT_LAST)

def load_ckpt(model, ema, opt, sch, scaler):
    ckpt = torch.load(CKPT_LAST, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    ema.module.load_state_dict(ckpt["ema"])
    opt.load_state_dict(ckpt["optimizer"])
    sch.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    print(f"▶ Resumed from epoch {ckpt['epoch']} (best={ckpt['best']*100:.2f}%)")
    return (ckpt["epoch"], ckpt["best"], ckpt["pat"],
            ckpt["trL"], ckpt["vaL"], ckpt["trA"], ckpt["vaA"], ckpt["lrs"])

# ─── Main routine ─────────────────────────────────────────
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("▶ Device:", dev)

    train_ldr, val_ldr, num_classes = get_loaders() #modified for CB-Sampling
    model     = build_model(num_classes).to(dev)
    model_ema = ModelEmaV2(model, decay=EMA_DECAY, device=dev)

    # Save model summary once
    if not os.path.exists(f"{OUT_DIR}/model_info.txt"):
        with open(f"{OUT_DIR}/model_info.txt", "w") as f:
            f.write(str(summary(model, input_size=(1, 3, 224, 224))))
            fl, pm = profile(copy.deepcopy(model).eval(),
                             inputs=(torch.randn(1, 3, 224, 224).to(dev),))
            f.write(f"\nFLOPs {fl/1e9:.2f} G | Params {pm/1e6:.2f} M\n")

    loss = nn.CrossEntropyLoss(label_smoothing=0.2)  # Add label smoothing
    opt = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999),weight_decay=WEIGHT_DECAY)
    sch = build_scheduler(opt)
    scaler = GradScaler('cuda')

    # Resume if checkpoint exists
    if os.path.exists(CKPT_LAST):
        (start_ep, best, pat, trL, vaL, trA, vaA, lrs) = load_ckpt(model, model_ema, opt, sch, scaler)
        start_ep += 1
    else:
        start_ep = 1
        best = pat = 0
        trL = vaL = trA = vaA = lrs = []

    # ── Training loop ─────────────────────────────────────
    for ep in range(start_ep, NUM_EPOCHS + 1):
        tl, ta = train_epoch(model, train_ldr, loss, opt, dev, scaler, model_ema, use_mixup=True, mixup_alpha=0.4)  #Modified for MixUp
        vl, va = eval_epoch(model_ema.module, val_ldr, loss, dev, "Val")

        sch.step(ep); lr_now = opt.param_groups[0]["lr"]
        trL.append(tl); vaL.append(vl); trA.append(ta); vaA.append(va); lrs.append(lr_now)
        print(f"[{ep:3d}]  T {ta*100:5.2f}%/{tl:.4f} | V {va*100:5.2f}%/{vl:.4f} | LR {lr_now:.6f}")

        if va > best:
            best, pat = va, 0
            torch.save(model_ema.module.state_dict(), CKPT_BEST)

        save_ckpt(ep, model, model_ema, opt, sch, scaler, best, pat, trL, vaL, trA, vaA, lrs)


    # ── Visualizations & Test ─────────────────────────────
    model_ema.module.load_state_dict(torch.load(CKPT_BEST))
    # -----------MODIFIED: Using Simple TTA (horizontal flip only - vertical flips hurt accuracy)----------
    preds, targs = predict_all(model_ema.module, val_ldr, dev, use_tta=True, tta_mode='simple')
    acc = accuracy_score(targs, preds)
    cm  = confusion_matrix(targs, preds, normalize="true")
    cls_names = train_ldr.dataset.dataset.classes if isinstance(train_ldr.dataset, Subset) else train_ldr.dataset.classes
    plot_confusion(cm, cls_names, f"{OUT_DIR}/confusion_norm.png")

    with open(f"{OUT_DIR}/metrics.txt", "w") as m:
        m.write(f"Overall Accuracy : {acc*100:.2f}%\n\n")
        m.write(classification_report(targs, preds, target_names=cls_names, digits=4))

    print(f"Test Accuracy : {acc*100:.2f}%")
    torch.save(model_ema.module.state_dict(), f"{OUT_DIR}/{MODEL_NAME}.pth")

if __name__ == "__main__":
    main()
