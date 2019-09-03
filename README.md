# Stronger Data Poisoning Attacks Break Data Sanitization Defenses

This code replicates the experiments from the following paper:

> Pang Wei Koh*, Jacob Steinhardt*, and Percy Liang
>
> [Stronger Data Poisoning Attacks Break Data Sanitization Defenses
](https://arxiv.org/abs/1811.00741)

## Dependencies
We use the following dependencies:
- Python 2.7.16
- gurobi for Python 7.0.2
- cvxpy 0.4.11 [note that cvxpy 1.0+ is not backwards compatible]
- numpy 1.16.2
- scikit-learn 0.20.3
- Tensorflow 1.12.0
- h5py 2.9.0
- MATLAB r2011b.

## Data and setup
For historical reasons, this codebase is split into Python and MATLAB components:
- The influence and KKT attacks are written in Python, as are the defenses and attack evaluation.
- The generation of the decoy parameters and the min-max attack are written in MATLAB.
The Python and MATLAB files only interact through shared input/output files.
All MATLAB files and dependencies are in the `matlab/` folder.

You can download the datasets we used [here](https://drive.google.com/open?id=1WvLITC-II9QBCU4E7CiIdVAJRAyi8ly0). For convenience, we have included .npz and .mat versions of the datasets for use in the Python and MATLAB files.

To set up the directory structure, you might want to edit the following files:
  `datasets.py`
    Set DATA_FOLDER to where the datasets/decoy parameters are stored
    and OUTPUT_FOLDER to where you'd like the attacks to be output.
    This affects all of the Python files and is independent of the MATLAB files.

  `pathdef.m`
    Needed in order to find gurobi, yalmip, sedumi, and the sever/svm subfolder.
    Will likely need to re-generate based on your local installation.

  `GRB_LICENSE_FILE`
    If you're using gurobi, this environmental variable needs to be set in your shell.

## Influence attack
Key files:
  `run_gradient_em_attack.py`
    To run the influence attack for a dataset using the same settings as we used in the paper, execute `python run_gradient_em_attack.py --em_iter 0 --total_grad_iter 10000 --dataset enron --use_slab --epsilon 0.03`, replacing 'enron' with the names of other datasets if needed.
    em_iter controls the number of iterative updates to the feasible set; we set this to 5000 to generate Figure 5.

  `copy_over_gradient_attacks.py`
    This is a helper file that converts the influence attack files into a similar format as the other attacks below.
    Execute, e.g., `python copy_over_gradient_attacks.py influence-em-0` or `python copy_over_gradient_attacks.py influence-em-5000` after the attack is finished.

  `influence/`
    Git repo for "Understanding Black-box Predictions via Influence Functions".
    We use helper functions from this repo to calculate influence.  

## Decoy parameter generation (for KKT and min-max attacks)
Key files:
  `generateThetaFnc.m`
    Generates target parameters, which are needed to run the KKT and min-max attacks.
    Calls generateTheta with the appropriate arguments.
    Uses pruneTheta to prune the set of generated theta down to a manageable amount.

## KKT attack
Key files:
  `run_kkt_attack.py`
    To run the kkt attack for a dataset using the same settings as we used in the paper, execute  `python run_kkt_attack.py kkt-loss-slab --dataset enron --repeat_points 2`, replacing 'enron' with the names of other datasets if needed.

## Min-max attack
Key files:
  `generateAttackFnc.m`
    Runs the min-max attack for a dataset using the same settings as we used in the paper. Assumes that generateThetaFnc has already been run with results saved to the locations specified in target_file.
    Calls generateAttackAll, which calls generateAttackTar (this final file is where most of the work of actually generating an attack occurs).
  `sever/`
    Git repo for "Sever: A Robust Meta-Algorithm for Stochastic Optimization".
    We make use of several helper functions for evaluating defenses and training models.

Other files:
  prettyPrint.m
    Helper file to pretty-print a struct
  randRound.m
    Helper file to randomly round a real-valued vector to an integer-valued vector

## Evaluating attacks against defenses
Key file:
  `test_defenses_for_all_epsilons.py`
    To evaluate an attack after generating it, run `python test_defenses_for_all_epsilons.py enron [attack_name] [--epsilon 0.03]`.
    attack_name can be, for example, 'influence-em-0', 'kkt-loss-slab-repeat2', or 'mm-loss-slab-train'.
