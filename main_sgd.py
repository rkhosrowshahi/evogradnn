import argparse
import numpy as np
import torch
import wandb
from tqdm import tqdm

from models import get_model
from utils import WarmUpLR, load_data, save_model, set_seed

print(torch.backends.mps.is_available()) #the MacOS is higher than 12.3+
print(torch.backends.mps.is_built()) #MPS is activated

# Fitness Function: Evaluate ResNet-18 on CIFAR-10
def evaluate_model_acc(model, data_loader, device, train=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if train:
                correct *= -1
                break
    accuracy = 100 * correct / total
    return accuracy

def evaluate_model_ce(model, data_loader, device, train=False):
    model.eval()
    loss = 0.0
    total_batch = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # outputs = torch.nan_to_num(outputs, nan=1e10)
            # outputs = torch.clamp(outputs, min=-1000, max=1000)
            loss += torch.nn.functional.cross_entropy(outputs, labels).item()
            total_batch += 1
            if train:
                break
    
    if not train:
        loss /= total_batch
    # print(loss)
    if np.isnan(loss) or np.isinf(loss):
        loss = 9.9e+21
    return loss

def train_on_gd(model, train_loader, optimizer, criterion, step=0, warmup_scheduler=None, device='cuda'):
    model.train()
    total_fe = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        loss.backward()
        optimizer.step()

        total_fe += 1

        if step <= args.warm and warmup_scheduler is not None:
            # print(f"Step {step}, Warmup Scheduler")
            warmup_scheduler.step()
            # print(f"lr: {optimizer.param_groups[0]['lr']}")

    return total_fe


def train_with_sgd(args):
    wandb.init(project=f"SGD-{args.dataset}-{args.net}", config={
        "steps": args.steps,
        "device": args.device,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "warm": args.warm,
        "seed": args.seed
    })
    set_seed(args.seed)
    evaluate = evaluate_model_ce
    # evaluate = evaluate_model_acc
    train_loader, val_loader, test_loader, num_classes = load_data(args.dataset, args.batch_size)
    # Initialize ResNet-18
    model = get_model(model_name=args.net, num_classes=num_classes).to(args.device)
    print(evaluate(model, train_loader, args.device, train=True))
    # total_weights = sum(p.numel() for p in model.parameters())
    # initial_weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    # train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    total_fe = 0
    # Training Loop
    state_wandb = {}
    
    pbar = tqdm(range(1, args.steps + 1), desc="Training")
    for step in pbar:
        if step > args.warm:
            train_scheduler.step()
        fe = train_on_gd(model, train_loader, optimizer, criterion, step=step, warmup_scheduler=warmup_scheduler, device=args.device)
        fitness = evaluate(model, train_loader, args.device, train=True)
        total_fe += fe

        if step % 10 == 0 or step == 1:
            val_accuracy = evaluate_model_acc(model, test_loader, device=args.device)

        state_wandb = {
                "test accuracy": val_accuracy,
                "step": step,
                "fitness": fitness,
                "function evaluations": total_fe,
                "LR": optimizer.param_groups[0]['lr']
            }
        wandb.log(state_wandb)
        pbar.set_postfix({"fitness": f"{fitness:.4f}", "fe": total_fe})

    # model = build_model(model, D, total_weights, mu, codebook, state, weight_offsets)
    # Final Evaluation
    final_accuracy = evaluate_model_acc(model, test_loader, device=args.device)
    print(f"Final Test Accuracy: {final_accuracy}%")
    wandb.log({"final_test_accuracy": final_accuracy})

    save_model(model, f"{args.dataset}_{args.net}_{args.lr}_{args.batch_size}_{args.steps}_{args.warm}", wandb)

    # Finish wandb run
    wandb.finish()

# Run Training
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--warm", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_with_sgd(args)