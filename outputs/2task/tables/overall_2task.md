# Overall performance comparison


## 2-task (mu, eps_LUMO)  (n_train=10000, seed=42)


### Overall

| Method | GNN dm% | GNN MR | GNN MR+STL | Uni-Mol dm% | Uni-Mol MR | Uni-Mol MR+STL |
|---|---|---|---|---|---|---|
| STL | ref | -- | 1.00 | ref | -- | 2.00 |
| LS | -27.1 | 3.00 | 4.00 | -7.2 | 3.50 | 4.50 |
| PCGrad | -29.8 | 4.00 | 5.00 | -1.2 | 1.00* | 1.50 |
| AIM-Scalar | -12.3 | 1.00* | 2.00 | -3.6 | 3.00 | 3.50 |
| AIM-Matrix | -17.7 | 2.00 | 3.00 | -3.4 | 2.50 | 3.50 |


### Per-task test MAE

| Backbone | Method | mu | eps_LUMO |
|---|---|---|---|
| GNN | STL | 0.1891 | 0.1353 |
| GNN | LS | 0.2693 | 0.1511 |
| GNN | PCGrad | 0.2777 | 0.1525 |
| GNN | AIM-Scalar | 0.2247 | 0.1430 |
| GNN | AIM-Matrix | 0.2373 | 0.1488 |
| Uni-Mol | STL | 0.1736 | 0.0862 |
| Uni-Mol | LS | 0.1872 | 0.0919 |
| Uni-Mol | PCGrad | 0.1847 | 0.0828 |
| Uni-Mol | AIM-Scalar | 0.1872 | 0.0856 |
| Uni-Mol | AIM-Matrix | 0.1852 | 0.0863 |


### Verdict

- GNN       best AIM = AIM-Scalar (MR 1.00)   best non-AIM = LS (MR 3.00)   -> AIM LEADS
- beats STL (dm% > 0): none
- Uni-Mol   best AIM = AIM-Matrix (MR 2.50)   best non-AIM = PCGrad (MR 1.00)   -> AIM does NOT lead
- beats STL (dm% > 0): none

