# [Active Learning for Physics Informed Neural Networks](https://github.com/carthurs/PINNs)

This repository contains software for an active learning approach to training physics-informed neural networks.

The core concept is that fluid domains and boundary conditions can be described parametrically, and networks can be trained on traditional finite element solution data, sparse in parameter space, and can interpolate those data in order to predict solutions between the training data. This means that solutions can be found in domain shapes - or with boundary conditions - which have never been simulated.

A key concept is the introduction of the active learning algorithm, whereby the training process iteratively determines which absent training data would be most informative to the network, and automatically generates a traditional finite element mesh for that domain, imposes those boundary conditions, and runs the simulation with a finite element solver, before injecting the results into the training set.

## Dependencies
* gmsh (tested with version 4.5.1)
* Nektar++ (tested with version 5.0.0)
* Paraview (optional - tested with version 5.7.0)

## Supporting Paper Abstract 

The goal of this work is to train a neural network which approximates solutions to the Navier-Stokes equations across a region of parameter space, in which the parameters define physical properties such as domain shape and boundary conditions. The contributions of this work are threefold:
1) To demonstrate that neural networks can be efficient aggregators of whole families of parameteric solutions to physical problems, trained using data created with traditional, trusted numerical methods such as finite elements. Advantages include extremely fast evaluation of pressure and velocity at any point in physical and parameter space (asymptotically, ~3 μs / query), and data compression (the network requires 99\% less storage space compared to its own training data).
2) To demonstrate that the neural networks can accurately interpolate between finite element solutions in parameter space, allowing them to be instantly queried for pressure and velocity field solutions to problems for which traditional simulations have never been performed.
3) To introduce an active learning algorithm, so that during training, a finite element solver can automatically be queried to obtain additional training data in locations where the neural network's predictions are in most need of improvement, thus autonomously acquiring and efficiently distributing training data throughout parameter space.
In addition to the obvious utility of Item 2, above, we demonstrate an application of the network in rapid parameter sweeping, very precisely predicting the degree of narrowing in a tube which would result in a 50\% increase in end-to-end pressure difference at a given flow rate. This capability could have applications in both medical diagnosis of arterial disease, and in computer-aided design. 

## Citation

ArXiv paper: _Active Training of Physics-Informed Neural Networks to Aggregate and Interpolate Parametric Solutions to the Navier-Stokes Equations_. Available here: https://arxiv.org/abs/2005.05092

    @misc{arthurs2020active,
    title={Active Training of Physics-Informed Neural Networks to Aggregate and Interpolate Parametric Solutions to the Navier-Stokes Equations},
    author={Christopher J Arthurs and Andrew P King},
    year={2020},
    eprint={2005.05092},
    archivePrefix={arXiv},
    primaryClass={physics.comp-ph}

## Key References
  - Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "[Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561)." arXiv preprint arXiv:1711.10561 (2017).
  
  - Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "[Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10566)." arXiv preprint arXiv:1711.10566 (2017).