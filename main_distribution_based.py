import argparse
import copy
import itertools
import jax
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from evosax.algorithms.distribution_based import distribution_based_algorithms
import wandb

from utils import WarmUpLR, build_model, evaluate_model_acc, evaluate_model_acc_single_batch, evaluate_model_ce, evaluate_model_ce_single_batch, evaluate_model_f1score_single_batch, load_data, save_model, set_seed, train_on_gd, ubp_cluster
from models import get_model

print(torch.backends.mps.is_available()) #the MacOS is higher than 12.3+
print(torch.backends.mps.is_built()) #MPS is activated


# Distribution based strategies Training Loop
def main(args):
    """
    Main training loop using distribution based strategies
    Args:
        args: Command line arguments
    """
    set_seed(args.seed)
    # Initialize wandb logging
    wandb.init(project=f"GF-{args.strategy}-{args.net}-{args.dataset}", config={
        "W_init": args.w_init,
        "steps": args.steps,
        "popsize": args.popsize,
        "device": args.device,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "net": args.net,
        "sigma_init": args.sigma_init,
        "lr_init": args.lr_init,
        "dataset": args.dataset,
        "weight_decay": args.weight_decay,
        "lr_decay": args.lr_decay,
        "eval_interval": args.eval_interval,
        "gd_interval": args.gd_interval,
        "objective": args.objective
    })

    # Setup evaluation function and device
    # obj = evaluate_model_ce
    obj = None
    if args.objective == 'acc':
        obj = evaluate_model_acc_single_batch
    elif args.objective == 'f1':
        obj = evaluate_model_f1score_single_batch  
    elif args.objective == 'ce':
        obj = evaluate_model_ce_single_batch
    else:
        raise ValueError(f"Objective {args.objective} not supported")
    # evaluate = evaluate_model_acc
    # Initialize ResNet-18
    device = args.device
    W_init = args.w_init
    popsize = args.popsize
    steps = args.steps
    
    train_loader, val_loader, test_loader, num_classes = load_data(args.dataset, args.batch_size)

    model = get_model(args.net, num_classes, device)
    gd_model = copy.deepcopy(model)

    # print(evaluate(model, train_loader, device, train=True))
    total_weights = sum(p.numel() for p in model.parameters())
    best_params = parameters_to_vector(model.parameters())

    codebook = {}
    # codebook = random_codebook_initialization(W_init, total_weights)
         
    weight_offsets = torch.normal(mean=0.0, std=1.0, size=(total_weights,), device=device)

    W = W_init
    D = W_init * 2
    # Initialize distribution based strategy
    rng = jax.random.PRNGKey(args.seed)
    x0 = np.concatenate([np.zeros(W_init), np.full(W_init, np.log(0.01))])
    solver = distribution_based_algorithms[args.strategy](population_size=popsize, solution=x0)
    es_params = solver.default_params
    if args.strategy == 'CMA_ES':
        es_params = es_params.replace(std_init=args.sigma_init)
    elif args.strategy == 'Sep_CMA_ES':
        es_params = es_params.replace(std_init=args.sigma_init)
    elif args.strategy == 'PGPE':
        es_params = es_params.replace(std_init=args.sigma_init,
                                      std_lr=args.std_lr)
    elif args.strategy == 'SimpleES':
        es_params = es_params.replace(std_init=args.sigma_init)
        
    state = solver.init(rng, x0, es_params)

    optimizer = torch.optim.SGD(gd_model.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    # train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.linspace(args.steps * 0.2, args.steps * 0.6, 3), gamma=0.2)
    # train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    val_accuracy = evaluate_model_acc(model, val_loader, device)
    test_accuracy = evaluate_model_acc(model, test_loader, device)
    total_fe = 0
    train_iter = itertools.cycle(train_loader)
    state_wandb = {}
    pbar = tqdm(range(1, args.steps + 1), desc=f"Training with {args.strategy}", disable=not args.verbose)
    for step in pbar:
        rng, rng_ask, rng_tell = jax.random.split(rng, 3)
        if step > args.warm:
            train_scheduler.step()
        val_accuracy_after_sgd = 0
        if step % args.gd_interval == 0 or step == 1:
            # model_instance = copy.deepcopy(model)
            vector_to_parameters(best_params, gd_model.parameters())
            gd_fe, gd_loss = train_on_gd(gd_model, train_loader, optimizer, criterion, step=step, warmup_scheduler=warmup_scheduler, args=args, device=device)
            val_accuracy_after_sgd = evaluate_model_acc(gd_model, val_loader, device)
            print(f"val_accuracy_after_sgd: {val_accuracy_after_sgd}, val_accuracy: {val_accuracy}")
            # if val_accuracy_after_sgd > val_accuracy:
            if True:
                best_params = parameters_to_vector(gd_model.parameters())
                del model
                model = copy.deepcopy(gd_model)
                
                codebook, centers, log_sigmas, assignment = ubp_cluster(W=W_init, params=best_params.detach().cpu().numpy())
                W = len(centers)
                D = W * 2
                x0 = np.concatenate([centers, log_sigmas])
                solver = distribution_based_algorithms[args.strategy](population_size=popsize, solution=x0.copy())
                es_params = es_params.replace(std_init=np.mean(state.std))
                state = solver.init(rng, mean=x0.copy(), params=es_params)
                val_accuracy = val_accuracy_after_sgd
            total_fe += gd_fe
    
        solutions, state = solver.ask(rng_ask, state, es_params)
        fitness_values = np.zeros(len(solutions))
        batch = next(train_iter)
        for i, x in enumerate(solutions):
            build_model(model, W, total_weights, x, codebook, state, weight_offsets)
            # Evaluate model
            # weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()
            # penalty = args.weight_decay * compute_l2_norm(weights)
            f = obj(model, batch, device, train=True) #+ penalty
            fitness_values[i] = f # Minimize 
        total_fe += popsize
        # Update distribution based strategy
        state, metrics = solver.tell(rng_tell, solutions, fitness_values, state, es_params)
        mu = solver.get_mean(state)
        build_model(model, W, total_weights, mu, codebook, state, weight_offsets)
        val_accuracy_after_ea = evaluate_model_acc(model, val_loader, device)
        print(f"val_accuracy_after_ea: {val_accuracy_after_ea}, val_accuracy: {val_accuracy}")
        if val_accuracy_after_ea > val_accuracy:
            val_accuracy = val_accuracy_after_ea
            best_params = parameters_to_vector(model.parameters())
        mean_fitness = obj(model, batch, device, train=True)
        min_fitness = np.min(fitness_values)
        average_fitness = np.mean(fitness_values)
        if step % args.eval_interval == 0:
            test_accuracy = evaluate_model_acc(model, test_loader, device)

        if np.mean(state.std) == 0  or np.isnan(np.mean(state.std)):
            print("sigma reached to 0, halting the optimization...")
            break

        # build_model(model, W, total_weights, solutions[np.argmin(fitness_values)], codebook, state, weight_offsets)
        mean_params = parameters_to_vector(model.parameters())
        _codebook, _centers, _log_sigmas, _assignment = ubp_cluster(W=W_init, params=mean_params.detach().cpu().numpy())
        x0 = np.concatenate([_centers, _log_sigmas])
        build_model(model, len(_centers), total_weights, x0, _codebook, state, weight_offsets)
        val_accuracy_after_ea_after_ubp = evaluate_model_acc(model, val_loader, device)
        print(f"val_accuracy_after_ea_after_ubp: {val_accuracy_after_ea_after_ubp}, val_accuracy: {val_accuracy}")
        if val_accuracy_after_ea_after_ubp >= val_accuracy_after_ea:
        # if True:
            W = len(_centers)
            D = W * 2
            codebook, centers, log_sigmas, assignment = _codebook, _centers, _log_sigmas, _assignment

            solver = distribution_based_algorithms[args.strategy](population_size=popsize, solution=x0.copy())
            es_params = es_params.replace(std_init=np.mean(state.std))
            state = solver.init(rng, mean=x0.copy(), params=es_params)

            # val_accuracy = val_accuracy_after_ea_after_ubp
        
        if val_accuracy_after_ea_after_ubp > val_accuracy:
            val_accuracy = val_accuracy_after_ea_after_ubp
            best_params = parameters_to_vector(model.parameters())


        state_wandb = {
                "test accuracy": test_accuracy,
                "step": step,
                "min fitness": min_fitness,
                "mean fitness": mean_fitness,
                "average fitness": average_fitness,
                "sigma": np.mean(state.std),
                "function evaluations": total_fe,
                "LR": optimizer.param_groups[0]['lr'],
                "D": D,
                "W": W
            }
        wandb.log(state_wandb)
        # Logging
        pbar.set_postfix({"D": D, "fitness": f"{min_fitness:.4f}", "fe": total_fe})


    # mu = solver.get_mean(state)
    # model = build_model(model, W, total_weights, mu, codebook, state, weight_offsets)
    vector_to_parameters(best_params, model.parameters())
    # Final Evaluation
    final_accuracy = evaluate_model_acc(model, test_loader, device)

    wandb.log({"final test accuracy": final_accuracy})

    save_model(model, f"GF-{args.strategy}-{args.net}-{args.dataset}-LR{args.lr_init}-W_init{args.w_init}-sigma_init{args.sigma_init}-batch_size{args.batch_size}-steps{args.steps}-warm{args.warm}", wandb)

    # Finish wandb run
    wandb.finish()

# Run Training
def parse_args():
    """
    Parse command line arguments
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR using distribution based strategies')
    
    # Training parameters
    parser.add_argument('--warm', type=int, default=1,
                      help='Warmup steps for SGD (default: 1)')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist'],
                      help='Dataset to use (cifar10 or cifar100 or mnist)')
    
    # Distribution based strategies parameters
    parser.add_argument('--strategy', type=str, default='CMA_ES', choices=list(distribution_based_algorithms.keys()),
                      help=f'Strategy to use {list(distribution_based_algorithms.keys())}')
    parser.add_argument('--w_init', type=int, default=256,
                      help='Codebook size (default: 2^8=256)')
    parser.add_argument('--steps', type=int, default=200,
                      help='Number of steps for distribution based strategy (default: 200)')
    parser.add_argument('--popsize', type=int, default=100,
                      help='Population size for distribution based strategy (default: 100)')
    parser.add_argument('--sigma_init', type=float, default=0.001,
                      help='Sigma initialization for distribution based strategy (default: 0.001)')
    parser.add_argument('--lr_init', type=float, default=0.1,
                      help='Learning rate for SGD (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                      help='Weight decay for SGD (default: 0.001)')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                      help='Learning rate decay for SGD (default: 0.9)')
    parser.add_argument('--eval_interval', type=int, default=10,
                      help='Evaluation interval for distribution based strategy (default: 100)')
    parser.add_argument('--gd_interval', type=int, default=10,
                      help='Evaluation interval for distribution based strategy (default: 100)')
    parser.add_argument('--objective', type=str, default='f1',
                      choices=['ce', 'acc', 'f1'],
                      help='Objective function to optimize (default: f1)')
    
    # Training parameters
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use for training (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Batch size for training (default: 128)')
    
    # Model parameters  
    parser.add_argument('--net', type=str, default='resnet18',
                      choices=['resnet18', 'resnet34', 'resnet50', 'mnist30k', 'mnist500k', 'mnist3m', 'cifar300k', 'cifar900k', 'cifar8m'],
                      help='ResNet model architecture')
    
    parser.add_argument('--verbose', default=False, action='store_true',
                      help='Verbose output (default: False)')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)