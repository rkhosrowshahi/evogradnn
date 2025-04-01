import argparse
import itertools
import jax
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import wandb
from evosax.algorithms.population_based import population_based_algorithms

from utils import *
from models import get_model

print(torch.backends.mps.is_available()) #the MacOS is higher than 12.3+
print(torch.backends.mps.is_built()) #MPS is activated

# Population-based Training Loop
def main(args):
    """
    Main training loop using population-based algorithms
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
        "lr_init": args.lr_init,
        "dataset": args.dataset,
        "weight_decay": args.weight_decay,
        "lr_decay": args.lr_decay,
        "eval_interval": args.eval_interval,
        "gd_interval": args.gd_interval,
        "objective": args.objective
    })

    # Setup evaluation function and device
    obj = evaluate_model_ce
    if args.objective == 'acc':
        obj = evaluate_model_acc_single_batch
    elif args.objective == 'f1':
        obj = evaluate_model_f1score_single_batch  
    elif args.objective == 'ce':
        obj = evaluate_model_ce_single_batch
    # evaluate = evaluate_model_acc
    # Initialize ResNet-18
    device = args.device
    W_init = args.w_init
    popsize = args.popsize
    steps = args.steps
    
    train_loader, val_loader, test_loader, num_classes = load_data(args.dataset, args.batch_size)

    model = get_model(args.net, num_classes, device)

    # print(evaluate(model, train_loader, device, train=True))
    total_weights = sum(p.numel() for p in model.parameters())
    gd_params = parameters_to_vector(model.parameters())

    codebook = {}
    # codebook = random_codebook_initialization(W_init, total_weights)
         
    weight_offsets = torch.normal(mean=0.0, std=1.0, size=(total_weights,), device=device)

    D = W_init * 2
    rng = jax.random.PRNGKey(args.seed)
    solver = population_based_algorithms[args.strategy](population_size=popsize, solution=np.zeros(D))  
    es_params = solver.default_params
    if args.strategy == 'DifferentialEvolution':
        es_params = es_params.replace(differential_weight=0.5, crossover_rate=0.9)
    elif args.strategy == 'DiffusionEvolution':
        es_params = es_params.replace(std_m=0.1)
    elif args.strategy == 'SimpleGA':
        es_params = es_params.replace(crossover_rate=0.5)
    elif args.strategy == 'SAMR_GA':
        es_params = es_params.replace(std_init=0.1)
    elif args.strategy == 'MR15_GA':
        es_params = es_params.replace(std_init=0.1)
    elif args.strategy == 'LGA':
        es_params = es_params.replace(std_init=0.1)
    elif args.strategy == 'GESMR_GA':
        es_params = es_params.replace(std_init=0.1)
    elif args.strategy == 'LearnedGA':
        es_params = es_params.replace(std_init=0.1)
    elif args.strategy == 'PSO':
        es_params = es_params.replace(inertia_coeff=0.7, cognitive_coeff=1.5, social_coeff=1.5)

    state = solver.init(rng, population=np.zeros((popsize, D)), fitness=np.inf, params=es_params)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.linspace(args.steps * 0.2, args.steps * 0.8, 3), gamma=0.2)
    # train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    val_accuracy = evaluate_model_acc(model, val_loader, device)
    total_fe = 0
    train_iter = itertools.cycle(train_loader)
    state_wandb = {}
    pbar = tqdm(range(1, args.steps + 1), desc=f"Training with {args.strategy}", disable=not args.verbose)
    for step in pbar:
        rng, rng_ask, rng_tell = jax.random.split(rng, 3)
        batch = next(train_iter)
        if step > args.warm:
            train_scheduler.step()
        
        val_accuracy_after_sgd = 0
        val_accuracy_after_ea = 0
        if step % args.gd_interval == 0 or step == 1:
            val_accuracy_after_ea = evaluate_model_acc(model, val_loader, device)
            if val_accuracy_after_ea < val_accuracy:
                vector_to_parameters(gd_params, model.parameters())

            # print("Before SGD fitness:")
            gd_fe, gd_loss = train_on_gd(model, train_loader, optimizer, criterion, step=step, warmup_scheduler=warmup_scheduler, args=args, device=device)
            # print("After SGD fitness:")
            val_accuracy_after_sgd = evaluate_model_acc(model, val_loader, device)
            if val_accuracy_after_sgd >= val_accuracy:
                gd_params = parameters_to_vector(model.parameters())
                codebook, centers, log_sigmas, assignment = ubp_cluster(W=W_init, params=gd_params.detach().cpu().numpy())
                W = len(centers)
                D = W * 2
                x0 = np.concatenate([centers, log_sigmas])
                build_model(model, W, total_weights, x0, codebook, state, weight_offsets)
                x0_fitness = obj(model, batch, device, train=True)
                # print(f"Initial fitness: {x0_fitness}")
                solver = population_based_algorithms[args.strategy](population_size=popsize, solution=x0)

                init_population = np.zeros((popsize, D))
                init_fitness = np.full(popsize, np.inf)
                for i in range(popsize):
                    # for j in range(W):
                    #     init_population[i][j] = np.random.uniform(centers[j] - np.exp(log_sigmas[j]), centers[j] + np.exp(log_sigmas[j]))
                    #     # init_population[i][j+W] = np.random.uniform(np.exp(log_sigmas[j]) - 0.1, np.exp(log_sigmas[j]) + 0.1)
                    #     init_population[i][j+W] = np.random.normal(0, 0.1)
                    init_population[i] = np.random.normal(x0, 0.001, size=D)
                    build_model(model, W, total_weights, init_population[i], codebook, state, weight_offsets)
                    weights = parameters_to_vector(model.parameters()).detach().cpu().numpy()
                    penalty = args.weight_decay * compute_l2_norm(weights)
                    f = obj(model, batch, device, train=True) + penalty
                    init_fitness[i] = f # Minimize 

                init_population[init_fitness.argmin()] = x0
                init_fitness[init_fitness.argmin()] = x0_fitness

                state = solver.init(rng, population=init_population, fitness=init_fitness, params=es_params)
            total_fe += gd_fe
            if step % args.eval_interval == 0 or step == 1:
                # val_accuracy_after_ea = evaluate_model_acc(model, val_loader, device)
                val_accuracy = val_accuracy_after_sgd if val_accuracy_after_sgd > val_accuracy_after_ea else val_accuracy_after_ea
    
        solutions, state = solver.ask(rng_ask, state, es_params)
        fitness_values = np.zeros(len(solutions))

        for i, solution in enumerate(solutions):
            model = build_model(model, W, total_weights, solution, codebook, state, weight_offsets)
            # Evaluate model
            
            weights = parameters_to_vector(model.parameters()).detach().cpu().numpy()
            penalty = args.weight_decay * compute_l2_norm(weights)
            f = obj(model, batch, device, train=True) + penalty
            fitness_values[i] = f # Minimize 
        total_fe += popsize
        # Update Population-based algorithm
        state, metrics = solver.tell(rng_tell, solutions, fitness_values, state, es_params)

        # best_solution = solver.get_best_solution(state)
        # build_model(model, W, total_weights, best_solution, codebook, state, weight_offsets)
        # min_fitness = obj(model, batch, device, train=True)
        min_fitness = state.fitness.min()
        
        mu = solver.get_population(state).mean(axis=0)
        build_model(model, W, total_weights, mu, codebook, state, weight_offsets)
        mean_fitness = obj(model, batch, device, train=True)
        average_fitness = state.fitness.mean()
        state_wandb = {
                "validation accuracy": val_accuracy,
                "step": step,
                "min fitness": min_fitness,
                "mean fitness": mean_fitness,
                "average fitness": average_fitness,
                "function evaluations": total_fe,
                "LR": optimizer.param_groups[0]['lr'],
                "D": D,
                "W": W
            }
        wandb.log(state_wandb)
        # Logging
        pbar.set_postfix({"D": D, "fitness": f"{min_fitness:.4f}", "fe": total_fe})
        # print(f"Step {step + 1}, D:{D}, Fitness: {np.min(fitness_values)}")


    mu = solver.get_population(state).mean(axis=0)
    model = build_model(model, W, total_weights, mu, codebook, state, weight_offsets)
    # Final Evaluation
    final_accuracy = evaluate_model_acc(model, test_loader, device)

    wandb.log({"final test accuracy": final_accuracy})

    save_model(model, f"GF-{args.strategy}-{args.net}-{args.dataset}-LR{args.lr_init}-W_init{args.w_init}-batch_size{args.batch_size}-steps{args.steps}-warm{args.warm}", wandb)

    # Finish wandb run
    wandb.finish()

# Run Training
def parse_args():
    """
    Parse command line arguments
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR using population-based algorithms')
    
    # Training parameters
    parser.add_argument('--warm', type=int, default=1,
                      help='Warmup steps for SGD (default: 1)')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist'],
                      help='Dataset to use (cifar10 or cifar100)')
    
    # Population-based algorithm parameters
    parser.add_argument('--strategy', type=str, default='DifferentialEvolution', choices=list(population_based_algorithms.keys()),
                      help='Population-based algorithm to use')
    parser.add_argument('--w_init', type=int, default=256,
                      help='Codebook size (default: 2^8=256)')
    parser.add_argument('--steps', type=int, default=200,
                      help='Number of steps for Population-based algorithm (default: 200)')
    parser.add_argument('--popsize', type=int, default=50,
                      help='Population size for Population-based algorithm (default: 50)')
    parser.add_argument('--lr_init', type=float, default=0.1,
                      help='Learning rate for SGD (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                      help='Weight decay for SGD (default: 0.001)')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                      help='Learning rate decay for SGD (default: 0.9)')
    parser.add_argument('--eval_interval', type=int, default=10,
                      help='Evaluation interval for Population-based algorithm (default: 10)')
    parser.add_argument('--gd_interval', type=int, default=10,
                      help='Evaluation interval for SGD (default: 10)')
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