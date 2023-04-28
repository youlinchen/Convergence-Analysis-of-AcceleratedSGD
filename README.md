# Convergence-Analysis-of-AcceleratedSGD 
## Introduction
This repository includes numrical experiments in Convergence Analysis of Accelerated StochasticGradient Descent under the Growth Condition (https://arxiv.org/abs/2006.06782). In particular, we conduct numerical experiments to validate our theoretical findings for the four considered accelerated methods: NAM, RMM, DAM+, and iDAM+. We optimize a simple quadratic loss, consider a stochastic oracle that satisfies the growth conditionbut not OUBV, and also implement SGD for the sake of a complete comparison. 

## Instruction:
1. Run each algirhtm by using `test_lower_bound.py`. For example, for fixing $\kappa=90$ and varying $\delta$, run
```
python test_lower_bound.py --L 90 --delta 0.6 --ratio 1e-8 --num_iter 2500 --num_trial 1000 --alg_name sgd --delta_range 0,2,100
python test_lower_bound.py --L 90 --delta 0.6 --ratio 1e-8 --num_iter 2500 --num_trial 1000 --alg_name nam --delta_range 0,1,100
python test_lower_bound.py --L 90 --delta 0.6 --ratio 1e-8 --num_iter 2000 --num_trial 1000 --alg_name rmm --delta_range 0,0.25,100
python test_lower_bound.py --L 90 --delta 0.6 --ratio 1e-8 --num_iter 2000 --num_trial 1000 --alg_name dam --delta_range 0,2,100
python test_lower_bound.py --L 90 --delta 0.6 --ratio 1e-8 --num_iter 2000 --num_trial 1000 --alg_name idam --delta_range 0,1,100
```
or for fixing $\delta=0.2$ and varying $\kappa$, run
```
python test_lower_bound.py --L 10 --delta 0.2 --ratio 1e-8 --num_iter 800 --num_trial 1000 --alg_name sgd --L_range 10,100,100
python test_lower_bound.py --L 10 --delta 0.2 --ratio 1e-8 --num_iter 800 --num_trial 1000 --alg_name nam --L_range 10,100,100
python test_lower_bound.py --L 10 --delta 0.2 --ratio 1e-8 --num_iter 800 --num_trial 1000 --alg_name rmm --L_range 10,100,100
python test_lower_bound.py --L 10 --delta 0.2 --ratio 1e-8 --num_iter 800 --num_trial 1000 --alg_name dam --L_range 10,100,100
python test_lower_bound.py --L 10 --delta 0.2 --ratio 1e-8 --num_iter 800 --num_trial 1000 --alg_name idam --L_range 10,100,100
```
2. Then use `summary.ipynb` to plot the figures. For example, the result of above commands is ![L_90](https://user-images.githubusercontent.com/26668349/235238493-c5b30883-1a33-494a-84cb-2c270708f044.png)
![delta_0 2](https://user-images.githubusercontent.com/26668349/235238505-21c64a98-6522-4c78-b732-cc54c16c36d0.png)
