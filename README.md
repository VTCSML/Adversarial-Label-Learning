# Adversarial-Label-Learning
This repository contains code for Adversarial Label Learning paper

If you use this work in an academic study, please cite our paper

```
@inproceedings{arachie2019adversarial,
  title={Adversarial label learning},
  author={Arachie, Chidubem and Huang, Bert},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={3183--3190},
  year={2019}
}
```

# Requirements

The library is tested in Python 3.6 and 3.7. Its main requirements are
scipy and numpy. Scikit-learn is also required to run the experiments


# Algorithm

The most important script is the train_classifier.py script that contains implementation of the algorithm. The other scripts are secondary classes for running experiments

# Examples

We have provided two classes for the experiments. Run_synthetic_experiment creates synthetic examples and shows the performance of the algorithm on synthetic data. Run_real_experiment runs experiments on the real datasets provided. Running experiment on other user datasets is fairly easy to implement


# Models

The model trained is a logistic regression classifier as reported in the paper. The train_classifier code can be modified to use more advanced classifier

# Bounds

The model is set to use the True bounds of the data. When this bounds is unknown, the user can provide an upper bounds for the weak signals or use constant bounds in the experiments scripts


# Limitations

The algorithm only supports binary classification and weak signals that do not abstain, we plan on providing code for ALL that fixes these limitations in the near future
