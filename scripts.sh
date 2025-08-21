
# for MNIST 
python distribution_based_trainer.py --dataset mnist --arch lenet --lr 0.1 --num_iterations 128 --epochs 100 --ws_device cuda --momentum 0.9 --d 32 --std 0.001 --ws_type mlpsoftsharing --wd 0

# for cifar10
python distribution_based_trainer.py --dataset cifar10 --arch resnet20 --lr 0.1 --num_iterations 128 --epochs 100 --ws_device cuda --momentum 0.9 --d 32 --std 0.001 --ws_type mlpsoftsharing --wd 0