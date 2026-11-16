# MLA03 – Reinforcement Learning Assignment 1

## Project structure
- Task01: RL framework notes
- Task02: MDP formulation is documented in the report
- Task03: Bellman one-step calculation program
- Task04: epsilon-greedy multi-armed bandit program
- Task05: fixed-time / greedy / epsilon-greedy simulation
- outputs: captured outputs from executed Python programs

## Requirements
- Python 3.10 or later
- NumPy
- python-docx (only required to regenerate the report)

Install:
```bash
python -m pip install numpy python-docx
```

Run:
```bash
python Task03/bellman_calculation.py
python Task04/epsilon_greedy_bandit.py
python Task05/traffic_comparison.py
```

## Reproducibility
The scripts use fixed random seeds. Task 4 follows the assignment's illustrative reward means [1.0, 1.5, 2.0, 1.2], Gaussian noise standard deviation 1.0, epsilon 0.1 and 1,000 decisions. Task 5 is a simplified bandit-style comparison with 100 episodes × 100 decisions. Its queue measure is an explicitly illustrative proxy, not a calibrated traffic simulator.

## Important
Replace the cover-page placeholders with your name, register number, submission date and GitHub URL. The PDF assignment requires progressive Git commits and output screenshots; create your own repository named `MLA03-<RegisterNumber>-Assignment1` and add these files.
