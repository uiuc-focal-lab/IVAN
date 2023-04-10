
# MNIST
# default eps = 0.3 works fine
python3 training_lirpa.py --bound_type IBP --model cnn_4layer --data MNIST 
python3 training_lirpa.py --bound_type IBP --model mlp_3layer --data MNIST 

python3 training_lirpa.py --bound_type CROWN-IBP --model cnn_4layer --data MNIST 
python3 training_lirpa.py --bound_type CROWN-IBP --model mlp_3layer --data MNIST 

python3 training_lirpa.py --bound_type CROWN-FAST --model cnn_4layer --data MNIST 
python3 training_lirpa.py --bound_type CROWN-FAST --model mlp_3layer --data MNIST 

# This is too slow
python3 training_lirpa.py --bound_type CROWN --model cnn_4layer --data MNIST 
python3 training_lirpa.py --bound_type CROWN --model mlp_3layer --data MNIST 

# CIFAR10
python3 training_lirpa.py --bound_type IBP --model cnn_4layer --data CIFAR --eps 0.007
python3 training_lirpa.py --bound_type IBP --model cnn_6layer --data CIFAR --eps 0.007

python3 training_lirpa.py --bound_type CROWN-IBP --model cnn_4layer --data CIFAR --eps 0.007
python3 training_lirpa.py --bound_type CROWN-IBP --model cnn_6layer --data CIFAR --eps 0.007

python3 training_lirpa.py --bound_type CROWN-FAST --model cnn_4layer --data CIFAR --eps 0.007
python3 training_lirpa.py --bound_type CROWN-FAST --model cnn_6layer --data CIFAR --eps 0.007

# This is too slow
python3 training_lirpa.py --bound_type CROWN --model cnn_4layer --data CIFAR --eps 0.007
python3 training_lirpa.py --bound_type CROWN --model cnn_6layer --data CIFAR --eps 0.007

# To test:
#  python3 analyzer.py --net training/IBP_CIFAR/cnn_4layer.onnx --dataset cifar10 --eps 0.03  --domain box
#  python3 analyzer.py --net training/IBP_CIFAR/cnn_4layer.onnx --dataset cifar10 --attack patch  --domain box

# Should also try --eps 0.03 for CIFAR?
# Should also train patch IBP?