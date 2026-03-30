# Suboptimality Loss for Inverse Learning from Imperfect Equilibria

Code for the networked Cournot competition experiments.

## Citation

```bibtex
@article{feik2026suboptimality,
  title   = {Suboptimality Loss for Inverse Learning from Imperfect Equilibria},
  author  = {Feik, Andreas and Pinson, Pierre and Paccagnan, Dario},
  year    = {2026},
  note    = {Working paper}
}
```

## Reproducing

```bash
pip install -r requirements.txt
python generate_data.py
python -m experiments.noise
python -m experiments.sample_size
python -m plotting.noise
python -m plotting.sample_size
```
