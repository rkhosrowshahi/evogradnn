import argparse
from ast import Pass
import itertools
import os
import sys
import jax
import numpy as np
import torch
from datetime import datetime

import wandb
from src.scheduler import *
from src.models import get_model
from src.utils import create_dataset, create_criterion, create_weight_sharing, population_based_strategy_init, distribution_based_strategy_init, evaluate_model_on_test, evaluate_solution_on_batch, evaluate_population_on_batch, load_solution_to_model, set_seed, params_to_vector, AverageMeter, STRATEGY_TYPES


def ea_train_epoch(optimizer, optimizer_params, optimizer_state, key, ws, criterion, train_loader, val_loader, epoch, step, learning_rate, device, args):
    """Train epoch for Evolutionary Algorithms (e.g., PSO)"""
    
    weight_decay = args.wd
    inner_steps = args.inner_steps

    train_loader_iterator = itertools.cycle(train_loader)
    num_batches = len(train_loader)
    period = num_batches // 10

    pop_mean_loss_meter = AverageMeter(name='pop_mean_loss', fmt=':.4e')
    pop_best_loss_meter = AverageMeter(name='pop_best_loss', fmt=':.4e')
    pop_avg_loss_meter = AverageMeter(name='pop_avg_loss', fmt=':.4e')

    velocity = torch.zeros(ws.D, device=device)

    for count_batch in range(1, num_batches+1):
        batch = next(train_loader_iterator)

        if args.optimizer.lower() == 'pso':
            population = optimizer.get_population(state=optimizer_state)
            fitnesses = evaluate_population_on_batch(population=population, ws=ws, criterion=criterion, batch=batch, device=device, weight_decay=weight_decay)
            optimizer_state = optimizer_state.replace(fitness=fitnesses, population_best=population, fitness_best=fitnesses, best_solution=population[np.argmin(fitnesses)], best_fitness=np.min(fitnesses))
        else:
            population = optimizer.get_population(state=optimizer_state)
            fitnesses = evaluate_population_on_batch(population=population, ws=ws, criterion=criterion, batch=batch, device=device, weight_decay=weight_decay)
            # print(f"Fitness avg on prev batch: {np.mean(optimizer_state.fitness)}, avg on curr batch: {np.mean(fitnesses)}, delta: {np.mean(fitnesses) - np.mean(optimizer_state.fitness)}")
            optimizer_state = optimizer_state.replace(fitness=fitnesses, best_solution=population[np.argmin(fitnesses)], best_fitness=np.min(fitnesses))

        for count_iter in range(1, inner_steps+1):
            # Get new solutions
            key, key_ask, key_tell = jax.random.split(key, 3)
            candidate, optimizer_state = optimizer.ask(key_ask, optimizer_state, optimizer_params)
            
            # Evaluate solutions
            candidate_fitnesses = evaluate_population_on_batch(population=candidate, ws=ws, criterion=criterion, batch=batch, device=device, weight_decay=weight_decay)
            # print(f"Best population fitness: {np.min(optimizer_state.fitness)}, candidate fitness: {np.min(candidate_fitnesses)}, delta: {np.min(candidate_fitnesses) - np.min(optimizer_state.fitness)}")
            candidate_fitness_argsort = np.argsort(candidate_fitnesses)
            topk_indices = candidate_fitness_argsort[:5]
            mean_candidate = candidate[topk_indices].mean(axis=0)
            mean_fitness = evaluate_solution_on_batch(z=mean_candidate, ws=ws, criterion=criterion, batch=batch, device=device)
            candidate = candidate.at[candidate_fitness_argsort[-1]].set(mean_candidate)
            candidate_fitnesses[candidate_fitness_argsort[-1]] = mean_fitness
            # Update ES
            optimizer_state, metrics = optimizer.tell(key=key_tell, population=candidate, fitness=candidate_fitnesses, state=optimizer_state, params=optimizer_params)

            pop_best_loss_meter.update(np.min(optimizer_state.fitness))
            pop_avg_loss_meter.update(np.mean(optimizer_state.fitness))

        # Extract signal from ES exploration
        mean_z = optimizer.get_population(state=optimizer_state).mean(axis=0)

        if args.bus.lower() == 'ema_in_es_loop':
            theta_base = ws.theta_base.clone()
            load_solution_to_model(mean_z, ws, device)
            theta_zt = params_to_vector(ws.model.parameters())
            delta = theta_zt - theta_base
            lr = args.lr
            velocity = args.momentum * velocity + (1 - args.momentum) * delta
            theta_base = theta_base + lr * velocity
            ws.set_theta(theta_base)
            # Align ES mean with new base theta without resetting ES state.
            # For linear mappings (randproj/sparseproj): theta = theta_base + alpha * P @ z
            # We compensate base shift Δtheta_base = lr * delta by Δz = - (alpha * P)^+ Δtheta_base
            try:
                if hasattr(ws, 'P') and ws.P is not None:
                    A = ws.alpha * ws.P  # (D, d)
                    delta_theta_base = (lr * velocity).to(A.device).float()  # (D,)
                    # Minimal-norm latent shift
                    pinvA = torch.linalg.pinv(A)  # (d, D)
                    delta_z_t = -(pinvA @ delta_theta_base)  # (d,)
                    delta_z_np = delta_z_t.detach().cpu().numpy()
                    # Update ES mean(s) directly to avoid losing momentum/state
                    mean_z = np.array(optimizer.get_population(state=optimizer_state).mean(axis=0))
                    if hasattr(optimizer_state, 'mean'):
                        optimizer_state = optimizer_state.replace(mean=mean_z + delta_z_np)
                    elif hasattr(optimizer_state, 'means'):
                        means_np = np.array(optimizer_state.means)
                        optimizer_state = optimizer_state.replace(means=means_np + delta_z_np[None, :])
            except Exception as e:
                print(f"Skipping Δz mean-translation due to error: {e}")

        # Log to wandb
        if (count_batch - 1) % period == 0 or count_batch == num_batches or count_batch == 1 or count_batch == 0:
            pop_mean_loss_meter.update(evaluate_solution_on_batch(z=mean_z, ws=ws, criterion=criterion, batch=batch, device=device))
            log_dict = {
                'Train/epoch': epoch,
                'Evolution/pop_std': np.std(optimizer.get_population(state=optimizer_state)),
                'Evolution/pop_best_loss': pop_best_loss_meter.avg,
                'Evolution/pop_avg_loss': pop_avg_loss_meter.avg,
                'Evolution/pop_mean_loss': pop_mean_loss_meter.avg,
                'Evolution/lr': learning_rate,
                'Evolution/step': step
            }
            if not args.disable_wandb:
                wandb.log(log_dict, step=step)

            print(f"Epoch {epoch}, batch {count_batch}, "
                    f"best pop loss: {pop_best_loss_meter.avg:.4f}, "
                    f"avg pop loss: {pop_avg_loss_meter.avg:.4f}, "
                    f"mean loss: {pop_mean_loss_meter.avg:.4f}")

        step += 1
    
    # print(f"best solution: {optimizer.get_best_solution(state=optimizer_state)[:5]}")
    return key, optimizer_state, step


def es_train_epoch(optimizer, optimizer_params, optimizer_state, key, ws, criterion, train_loader, val_loader, epoch, step, learning_rate, device, args):
    """Train epoch for Evolution Strategies (e.g., CMA-ES, Open-ES)"""
    
    weight_decay = args.wd
    inner_steps = args.inner_steps

    train_loader_iterator = itertools.cycle(train_loader)
    num_batches = len(train_loader)
    period = num_batches // 10

    mean_loss_meter = AverageMeter(name='mean_loss', fmt=':.4e')
    pop_best_loss_meter = AverageMeter(name='pop_best_loss', fmt=':.4e')
    pop_avg_loss_meter = AverageMeter(name='pop_avg_loss', fmt=':.4e')

    velocity = torch.zeros(ws.D, device=device)

    for count_batch in range(1, num_batches+1):
        batch = next(train_loader_iterator)

        for count_iter in range(1, inner_steps+1):
            # Get new solutions
            key, key_ask, key_tell = jax.random.split(key, 3)
            population, optimizer_state = optimizer.ask(key_ask, optimizer_state, optimizer_params)
            
            # Evaluate solutions
            fitnesses = evaluate_population_on_batch(population=population, ws=ws, criterion=criterion, batch=batch, device=device, weight_decay=weight_decay)

            # Update ES
            optimizer_state, metrics = optimizer.tell(key=key_tell, population=population, fitness=fitnesses, state=optimizer_state, params=optimizer_params)

            pop_best_loss_meter.update(np.min(fitnesses))
            pop_avg_loss_meter.update(np.mean(fitnesses))

        # Extract signal from ES exploration
        mean_z = optimizer.get_mean(state=optimizer_state)
        if args.optimizer.lower() == 'sv_cma_es' or args.optimizer.lower() == 'sv_open_es':
            mean_z = np.array(mean_z.mean(axis=0))

        if args.bus.lower() == 'ema_in_es_loop':
            theta_base = ws.theta_base.clone()
            load_solution_to_model(mean_z, ws, device)
            theta_zt = params_to_vector(ws.model.parameters())
            delta = theta_zt - theta_base
            lr = args.lr
            velocity = args.momentum * velocity + (1 - args.momentum) * delta
            theta_base = theta_base + lr * velocity
            ws.set_theta(theta_base)
            # Align ES mean with new base theta without resetting ES state.
            # For linear mappings (randproj/sparseproj): theta = theta_base + alpha * P @ z
            # We compensate base shift Δtheta_base = lr * delta by Δz = - (alpha * P)^+ Δtheta_base
            try:
                if hasattr(ws, 'P') and ws.P is not None:
                    A = ws.alpha * ws.P  # (D, d)
                    delta_theta_base = (lr * velocity).to(A.device).float()  # (D,)
                    # Minimal-norm latent shift
                    pinvA = torch.linalg.pinv(A)  # (d, D)
                    delta_z_t = -(pinvA @ delta_theta_base)  # (d,)
                    delta_z_np = delta_z_t.detach().cpu().numpy()
                    # Update ES mean(s) directly to avoid losing momentum/state
                    mean_z = np.array(optimizer.get_mean(state=optimizer_state).mean(axis=0))
                    if hasattr(optimizer_state, 'mean'):
                        optimizer_state = optimizer_state.replace(mean=mean_z + delta_z_np)
                    elif hasattr(optimizer_state, 'means'):
                        means_np = np.array(optimizer_state.means)
                        optimizer_state = optimizer_state.replace(means=means_np + delta_z_np[None, :])
            except Exception as e:
                print(f"Skipping Δz mean-translation due to error: {e}")

        # Log to wandb
        if (count_batch - 1) % period == 0 or count_batch == num_batches or count_batch == 1 or count_batch == 0:
            mean_loss_meter.update(evaluate_solution_on_batch(z=mean_z, ws=ws, criterion=criterion, batch=batch, device=device))
            log_dict = {
                'Evolution/es_sigma': np.mean(optimizer_state.std),
                'Evolution/pop_best_loss': pop_best_loss_meter.avg,
                'Evolution/pop_avg_loss': pop_avg_loss_meter.avg,
                'Evolution/mu_loss': mean_loss_meter.avg,
                'Evolution/lr': learning_rate,
                'Evolution/step': step
            }
            if not args.disable_wandb:
                wandb.log(log_dict, step=step)

            print(f"Epoch {epoch}, batch {count_batch}, "
                    f"best pop loss: {pop_best_loss_meter.avg:.4f}, "
                    f"avg pop loss: {pop_avg_loss_meter.avg:.4f}, "
                    f"mean loss: {mean_loss_meter.avg:.4f}")

        step += 1
    
    return key, optimizer_state, step


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check CUDA memory and provide information
    if device == 'cuda':
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        torch.cuda.empty_cache()  # Clear any existing cache
    else:
        print("Using CPU device")
    
    key = jax.random.PRNGKey(0)
    if args.seed is not None:
        key = jax.random.PRNGKey(args.seed)
        set_seed(args.seed)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create save directory
    save_path = os.path.join(args.save_path, timestamp)
    os.makedirs(save_path, exist_ok=True)
    print(f"Created save directory: {save_path}")
    os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)

    print(f"Starting training with arguments: {args}")
    
    # Load data and model
    train_loader, val_loader, test_loader, num_classes, input_size = create_dataset(args, validation_split=0.01)

    model = get_model(args.arch, input_size, num_classes, device)
    theta_0 = params_to_vector(model.parameters())
    num_weights = len(theta_0)
    
    # Setup loss function
    criterion = create_criterion(args=args, num_classes=num_classes)

    optimizer_type = STRATEGY_TYPES[args.optimizer.lower()]
    
    ws = create_weight_sharing(model=model, args=args, optimizer_type=optimizer_type)

    d = ws.d
    args.d = d
    
    x0 = np.zeros(d)

    # Determine which strategy to use and initialize accordingly
    if optimizer_type == 'EA':
        optimizer, optimizer_params = population_based_strategy_init(strategy=args.optimizer, args=args, x0=x0, steps=len(train_loader) * args.epochs)
        # Initialize population state
        # init_population = np.random.normal(x0, args.ea_std, size=(args.popsize, d))
        init_population = np.random.uniform(-args.ea_std, args.ea_std, size=(args.popsize, d))
        init_population[0] = x0.copy()
        init_fitness = evaluate_population_on_batch(population=init_population, ws=ws, criterion=criterion, batch=next(iter(train_loader)), device=device, weight_decay=args.wd)
        optimizer_state = optimizer.init(key=key, population=init_population, fitness=init_fitness, params=optimizer_params)
        train_epoch_fn = ea_train_epoch
    elif optimizer_type == 'ES':  # ES strategy
        optimizer, optimizer_params, optimizer_state = distribution_based_strategy_init(key=key, strategy=args.optimizer, x0=x0, steps=args.inner_steps * len(train_loader) * args.epochs, args=args)
        train_epoch_fn = es_train_epoch

    if args.lr_scheduler.lower() == 'cosine':
        scheduler = CosineAnnealingLRScheduler(eta_max=args.lr, eta_min=1e-4, T_max=args.epochs, T_mult=1)
    elif args.lr_scheduler.lower() == 'step':
        scheduler = StepLRScheduler(args.lr, args.lr_scheduler_step_size, args.lr_scheduler_gamma)
    elif args.lr_scheduler.lower() == 'multi_step':
        scheduler = MultiStepLRScheduler(args.lr, args.lr_scheduler_milestones, args.lr_scheduler_gamma)
    elif args.lr_scheduler.lower() == 'constant':
        scheduler = ConstantLRScheduler(args.lr)

    # Initialize wandb
    if not args.disable_wandb:
        config = {
            'timestamp': timestamp,
            'strategy': optimizer_type,
            'arch': args.arch,
            'dataset': args.dataset,
            'sampler': args.sampler,
            'optimizer': args.optimizer,
            'weight_sharing': args.ws,
            'normalize_projection': args.normalize_projection,
            'base_update_strategy': args.bus,
            'criterion': args.criterion,
            'f1_temperature': args.f1_temperature,
            'f1_learnable_temperature': args.f1_learnable_temperature,
            'f1_beta': args.f1_beta,
            'ce_weight': args.ce_weight,
            'f1_weight': args.f1_weight,
            'label_smoothing': args.label_smoothing,
            'ce_normalize': args.ce_normalize,
            'seed': args.seed,
            'epochs': args.epochs,
            'inner_steps': args.inner_steps,
            'dimensions': args.d,
            'popsize': args.popsize,
            'ea_std': args.ea_std if optimizer_type == 'EA' else None,
            'es_std': args.es_std if optimizer_type == 'ES' else None,
            'lr': args.lr,
            'lr_scheduler': args.lr_scheduler,
            'lr_scheduler_step_size': args.lr_scheduler_step_size,
            'lr_scheduler_gamma': args.lr_scheduler_gamma,
            'lr_scheduler_milestones': args.lr_scheduler_milestones,
            'weight_decay': args.wd,
            'momentum': args.momentum,
            'batch_size': args.batch_size,
            'parameters': num_weights,
            'hidden_dims': args.hidden_dims,
            'use_activation': args.use_activation,
            'activation': args.activation,
            'notes': args.note,
        }
        run_name = args.wandb_name if args.wandb_name else f"{timestamp}"
        
        wandb.init(
            project=f"{args.wandb_project}-{args.dataset}-{args.arch}-v2",
            name=run_name,
            config=config,
            tags=[args.arch, args.dataset, args.optimizer, args.ws, args.bus, optimizer_type]
        )

    # Evaluate theta_base as initial random parameters theta_0
    theta_base = theta_0.clone()
    ws.load_to_model(theta_base)
    theta_base_test_ce, theta_base_test_top1, theta_base_test_top5, theta_base_test_f1 = evaluate_model_on_test(model=ws.model, 
                                                            data_loader=test_loader, 
                                                            device=device, 
                                                            train=False)

    velocity = torch.zeros(num_weights, device=device)
    # Main optimization loop
    best_acc, best_loss = 0.0, 0.0
    epoch = 1
    step = 1
    while epoch <=  args.epochs:
        lr = scheduler.get_lr()
        if epoch <= 1:
            lr = 1.0 # warmup

        # Use the appropriate train_epoch function based on strategy
        key, optimizer_state, step = train_epoch_fn(optimizer, optimizer_params, optimizer_state, key, 
                                    ws, criterion, 
                                    train_loader, val_loader, 
                                    epoch, step, lr, 
                                    device, args)

        # Get mean solution from EA or ES
        if optimizer_type == 'EA':
            population = optimizer.get_population(state=optimizer_state)
            topk_indices = np.argsort(optimizer_state.fitness)[:5]
            zt = population[topk_indices].mean(axis=0)
            # zt = optimizer.get_best_solution(state=optimizer_state)
        elif optimizer_type == 'ES':  # ES strategy
            zt = optimizer.get_mean(state=optimizer_state)
            if args.optimizer.lower() == 'sv_cma_es' or args.optimizer.lower() == 'sv_open_es':
                zt = np.array(zt.mean(axis=0))
        
        load_solution_to_model(zt, ws, device)
        theta_t = params_to_vector(ws.model.parameters())

        theta_t_test_ce, theta_t_test_top1, theta_t_test_top5, theta_t_test_f1 = evaluate_model_on_test(model=ws.model, 
                                                            data_loader=test_loader, 
                                                            device=device, 
                                                            train=False)
        print(f"theta_base at epoch {epoch-1}: {theta_base[:5]}")
        print(f"theta_t at epoch {epoch}: {theta_t[:5]}")
        delta = theta_t - theta_base
        delta_norm = torch.norm(delta)
        print(f"delta at epoch {epoch}: {delta[:5]}")
        test_ce, test_top1, test_top5, test_f1 = 0, 0, 0, 0
        
        if args.bus.lower() == 'full':
            theta_base = theta_t.clone()
            ws.load_to_model(theta_base)
            (theta_base_test_ce, 
            theta_base_test_top1, 
            theta_base_test_top5, 
            theta_base_test_f1) = evaluate_model_on_test(model=ws.model, 
                                                        data_loader=test_loader, 
                                                        device=device, 
                                                        train=False)
            test_ce, test_top1, test_top5, test_f1 = theta_base_test_ce, theta_base_test_top1, theta_base_test_top5, theta_base_test_f1
            ws.set_theta(theta_base)
            ws.init()
            
            if optimizer_type == 'EA':
                init_population = np.random.normal(x0, args.ea_std, size=(args.popsize, d))
                optimizer_state = optimizer.init(key=key, population=init_population, fitness=np.inf * np.ones(args.popsize), params=optimizer_params)
            elif optimizer_type == 'ES':  # ES
                optimizer_state = optimizer.init(key=key, mean=np.zeros(args.d), params=optimizer_params)
                
        elif args.bus.lower() == 'ema':
            theta_base = theta_base + lr * delta
            ws.load_to_model(theta_base)
            (theta_base_test_ce, 
            theta_base_test_top1, 
            theta_base_test_top5, 
            theta_base_test_f1) = evaluate_model_on_test(model=ws.model, 
                                                        data_loader=test_loader, 
                                                        device=device, 
                                                        train=False)
            test_ce, test_top1, test_top5, test_f1 = theta_base_test_ce, theta_base_test_top1, theta_base_test_top5, theta_base_test_f1
            ws.set_theta(theta_base)
            ws.init()
            
            if optimizer_type == 'EA':
                init_population = np.random.normal(x0, args.ea_std, size=(args.popsize, d))
                optimizer_state = optimizer.init(key=key, population=init_population, fitness=np.inf * np.ones(args.popsize), params=optimizer_params)
            elif optimizer_type == 'ES':  # ES
                optimizer_state = optimizer.init(key=key, mean=np.zeros(args.d), params=optimizer_params)
                
        elif args.bus.lower() == 'ema_with_momentum':
            velocity = args.momentum * velocity + (1 - args.momentum) * delta
            theta_base = theta_base + lr * velocity
            ws.load_to_model(theta_base)
            (theta_base_test_ce, 
            theta_base_test_top1, 
            theta_base_test_top5, 
            theta_base_test_f1) = evaluate_model_on_test(model=ws.model, 
                                                        data_loader=test_loader, 
                                                        device=device, 
                                                        train=False)
            test_ce, test_top1, test_top5, test_f1 = theta_base_test_ce, theta_base_test_top1, theta_base_test_top5, theta_base_test_f1
            ws.set_theta(theta_base)
            ws.init()
            
            if optimizer_type == 'EA':
                init_population = np.random.normal(x0, args.ea_std, size=(args.popsize, d))
                optimizer_state = optimizer.init(key=key, population=init_population, fitness=np.inf * np.ones(args.popsize), params=optimizer_params)
            elif optimizer_type == 'ES':  # ES
                optimizer_state = optimizer.init(key=key, mean=np.zeros(args.d), params=optimizer_params)
                
        elif args.bus.lower() == 'ema_with_init_optimizer_state_and_update_mean':
            theta_base = theta_base + lr * delta
            ws.load_to_model(theta_base)
            (theta_base_test_ce, 
            theta_base_test_top1, 
            theta_base_test_top5, 
            theta_base_test_f1) = evaluate_model_on_test(model=ws.model, 
                                                        data_loader=test_loader, 
                                                        device=device, 
                                                        train=False)
            test_ce, test_top1, test_top5, test_f1 = theta_base_test_ce, theta_base_test_top1, theta_base_test_top5, theta_base_test_f1
            ws.set_theta(theta_base)
            # Align ES mean with new base theta without resetting ES state.
            # For linear mappings (randproj/sparseproj): theta = theta_base + alpha * P @ z
            # We compensate base shift Δtheta_base = lr * delta by Δz = - (alpha * P)^+ Δtheta_base
            try:
                if hasattr(ws, 'P') and ws.P is not None:
                    A = ws.alpha * ws.P  # (D, d)
                    delta_theta_base = (lr * delta).to(A.device).float()  # (D,)
                    # Minimal-norm latent shift
                    pinvA = torch.linalg.pinv(A)  # (d, D)
                    delta_z_t = -(pinvA @ delta_theta_base)  # (d,)
                    delta_z_np = delta_z_t.detach().cpu().numpy()
                    
                    if optimizer_type == 'EA':
                        init_population = np.random.normal(x0, args.ea_std, size=(args.popsize, d))
                        optimizer_state = optimizer.init(key=key, population=init_population, fitness=np.inf * np.ones(args.popsize), params=optimizer_params)
                    elif optimizer_type == 'ES':  # ES
                        optimizer_state = optimizer.init(key=key, mean=zt + delta_z_np, params=optimizer_params)
            except Exception as e:
                print(f"Skipping Δz mean-translation due to error: {e}")
                
        elif args.bus.lower() == 'none':
            test_ce, test_top1, test_top5, test_f1 = theta_t_test_ce, theta_t_test_top1, theta_t_test_top5, theta_t_test_f1
        else:
            theta_base = ws.theta_base
            ws.load_to_model(theta_base)
            (theta_base_test_ce, 
            theta_base_test_top1, 
            theta_base_test_top5, 
            theta_base_test_f1) = evaluate_model_on_test(model=ws.model, 
                                                        data_loader=test_loader, 
                                                        device=device, 
                                                        train=False)
            test_ce, test_top1, test_top5, test_f1 = theta_base_test_ce, theta_base_test_top1, theta_base_test_top5, theta_base_test_f1

        print(f"theta_base at epoch {epoch}: {theta_base[:5]}")

        log_dict = {
                     'Epoch': epoch,
                     'Train/lr': lr,
                     'Train/epoch': epoch,
                     'Test/ce': test_ce,
                     'Test/top1': test_top1,
                     'Test/top5': test_top5,
                     'Test/f1': test_f1,
                     'Test/theta_t_ce': theta_t_test_ce,
                     'Test/theta_t_top1': theta_t_test_top1,
                     'Test/theta_t_top5': theta_t_test_top5,
                     'Test/theta_t_f1': theta_t_test_f1,
                     'Test/theta_base_ce': theta_base_test_ce,
                     'Test/theta_base_top1': theta_base_test_top1,
                     'Test/theta_base_top5': theta_base_test_top5,
                     'Test/theta_base_f1': theta_base_test_f1,
                     'Evolution/delta_norm': delta_norm,
            }
        if not args.disable_wandb:
            wandb.log(log_dict, step=step)

        print(f"Epoch {epoch}, theta_base test loss: {theta_base_test_ce:.3f}, theta_base test top1: {theta_base_test_top1:.3f}, theta_t test loss: {theta_t_test_ce:.3f}, theta_t test top1: {theta_t_test_top1:.3f}")

        if best_acc < test_top1:
            best_acc = test_top1
            best_loss = test_ce
            print(f"New best top1: {best_acc:.3f}")
            torch.save({
                'state_dict': model.state_dict()
            }, os.path.join(save_path, "checkpoints", f"best_checkpoint.pth.tar"))
            print(f"Saved checkpoint to {os.path.join(save_path, 'checkpoints', f'best_checkpoint.pth.tar')}")

        epoch += 1

        scheduler.step()
    
    torch.save({
                'state_dict': model.state_dict()
            }, os.path.join(save_path, "checkpoints", f"final_checkpoint.pth.tar"))
    print(f"Saved final checkpoint to {os.path.join(save_path, 'checkpoints', f'final_checkpoint.pth.tar')}")
    log_dict = {
        'Final/test_top1': best_acc,
        'Final/test_ce': best_loss,
    }
    if not args.disable_wandb:
        wandb.log(log_dict, step=step)
        wandb.finish()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified Evolutionary Neural Network Training with Weight Sharing')

    
    # ============================================================================
    # Model and Dataset Configuration
    # ============================================================================
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (None for random seed)')
    parser.add_argument('--arch', type=str, default='resnet32',
                       help='Neural network architecture to use (e.g., resnet32, vgg16)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset to train on (e.g., cifar10, cifar100, mnist)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--val_split', type=float, default=0.01,
                       help='Validation split for training')
    parser.add_argument('--sampler', type=str, default=None,
                       help='Sampler for training')
    parser.add_argument('--criterion', type=str, default='ce', choices=['ce', 'mse', 'f1', 'soft_f1', 'ce_sf1'],
                       help='Loss function: ce (cross-entropy), f1 (F1 score), soft_f1 (soft F1), or ce_sf1 (CE + soft F1)')
    
    # ============================================================================
    # Training Hyperparameters
    # ============================================================================
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--f1_temperature', '--temperature', type=float, default=1.0,
                       help='Temperature for soft F1 score error')
    parser.add_argument('--f1_beta', type=float, default=0.0,
                       help='Beta for soft F1 score error')
    parser.add_argument('--f1_learnable_temperature', type=bool, default=False,
                       help='Learnable temperature for soft F1 score error')
    parser.add_argument('--ce_weight', type=float, default=0.5,
                       help='Weight for CrossEntropy loss in combined loss (default: 0.5)')
    parser.add_argument('--ce_normalize', type=str, default='none', choices=['none', 'log', 'minmax', 'zscore'],
                       help='Normalization method for CrossEntropy loss')
    parser.add_argument('--f1_weight', type=float, default=0.5,
                       help='Weight for F1 loss in combined loss (default: 0.5)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing for soft F1 score error, use 0.1 for CIFAR-10 and 0.05 for CIFAR-100 datasets')

    # ============================================================================
    # Optimizer and Learning Rate Configuration
    # ============================================================================
    parser.add_argument('--lr', "--learning_rate", type=float, default=1.0,
                       help='Initial learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='constant', 
                       choices=['cosine', 'step', 'multi_step', 'constant'],
                       help='Learning rate scheduler type')
    parser.add_argument('--lr_scheduler_step_size', type=int, default=10,
                       help='Step size for step scheduler')
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.1,
                       help='Gamma (decay factor) for step/multi_step schedulers')
    parser.add_argument('--lr_scheduler_milestones', type=str, default=None,
                       help='Milestones for multi_step lr scheduler (comma-separated)')
    parser.add_argument('--wd', "--weight_decay", type=float, default=5e-4,
                       help='Weight decay (L2 regularization) coefficient')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum factor for SGD optimizer')
    
    # ============================================================================
    # Evolutionary Strategy Configuration
    # ============================================================================
    parser.add_argument('--optimizer', type=str, default='PSO',
                       help='Evolutionary optimizer to use (EA: PSO, etc. | ES: CMA_ES, SV_CMA_ES, SimpleES, Open_ES, SV_Open_ES, xNES)')
    parser.add_argument('--inner_steps', type=int, default=1,
                       help='Number of inner optimization steps per iteration')
    parser.add_argument('--d', "--dimensions", type=int, default=128,
                       help='Dimensionality of the search space')
    parser.add_argument('--popsize', type=int, default=50,
                       help='Population size for evolutionary strategy')
    parser.add_argument('--ea_std', type=float, default=1.0,
                       help='Standard deviation for EA noise generation (used when strategy=ea)')
    parser.add_argument('--es_std', type=float, default=0.1,
                       help='Standard deviation for ES noise generation (used when strategy=es)')
    
    # ============================================================================
    # Weight Sharing Configuration
    # ============================================================================
    parser.add_argument('--ws', "--weight_sharing", type=str, default='randproj',
                       help='Weight sharing strategy (e.g., randproj, mlp, hard)')
    parser.add_argument('--bus', type=str, default='none',
                       help='Base update strategy (BUS) for weight sharing: none, full, ema, ema_with_momentum, ema_with_init_optimizer_state_and_update_mean, or ema_in_es_loop')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Scaling factor for random projection weight sharing')
    parser.add_argument('--normalize_projection', action='store_true',
                       help='Normalize projection matrix for random projection weight sharing')
    parser.add_argument('--use_activation', action='store_true',
                       help='Use activation function in MLP soft sharing')
    parser.add_argument('--activation', type=str, default=None,
                       choices=['relu', 'tanh', 'gelu', 'leaky_relu'],
                       help='Activation function for MLP weight sharing')
    parser.add_argument('--hidden_dims', type=str, default=None,
                       help='Hidden dimensions for MLP soft sharing (comma-separated)')
    parser.add_argument('--ws_device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device for weight sharing computations')
    
    # ============================================================================
    # Logging and Output Configuration
    # ============================================================================
    parser.add_argument('--save_path', type=str, default='logs/results',
                       help='Directory to save training results and checkpoints')
    
    # ============================================================================
    # Weights & Biases (wandb) Configuration
    # ============================================================================
    parser.add_argument('--wandb_project', type=str, default='evows',
                       help='Wandb project name for experiment tracking')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Wandb run name (auto-generated if None)')
    parser.add_argument('--disable_wandb', action='store_true',
                       help='Disable wandb logging completely')
    parser.add_argument('--note', type=str, default=None,
                       help='Additional note/description for the experiment run')

    args = parser.parse_args()
    main(args)

