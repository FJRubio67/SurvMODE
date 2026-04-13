# SurvMODE

## Survival Modelling via Ordinary Differential Equations

## Overview

This repository contains the R and Julia code used to reproduce the two case
studies in:

> Christen, J.A. and Rubio, F.J. (2026). Hazard-based distributional regression
> via ordinary differential equations.
> *Statistical Methods in Medical Research*, in press.
> [DOI: 10.1177/09622802251412840](https://doi.org/10.1177/09622802251412840) |
> [Preprint (arXiv)](https://arxiv.org/abs/2512.16336)

Traditional hazard-based survival models specify the baseline hazard through a
parametric distribution. This paper proposes an alternative approach in which the
hazard function is defined implicitly as the solution to an ordinary differential
equation (ODE), enabling flexible distributional regression for survival data.
The framework is illustrated through two real-data case studies and supports both
maximum likelihood (R) and Bayesian posterior sampling (Julia) inference.

## Repository structure

```
SurvMODE/
├── R code/
│   ├── Example1_2A.html     # Rendered tutorial with code and outputs (Case Study I)
│   └── ...                  # R scripts for Case Study I
└── Julia code/
    ├── rotterdamJulia.html  # Rendered tutorial with code and outputs (Case Study II)
    └── ...                  # Julia scripts for Case Study II
```

## Case studies

| | Case Study I | Case Study II |
|---|---|---|
| **Language** | R | Julia |
| **Inference** | MLE and Bayesian | MLE and Bayesian |
| **Dataset** | Ipilimumab data | Rotterdam breast cancer data |
| **Tutorial** | `Example1_2A.html` | `rotterdamJulia.html` |

## Requirements

**R (Case Study I).** Install the required packages by inspecting the `library()`
calls at the top of the R scripts. Key dependencies are likely to include:

```r
install.packages(c("survival", "deSolve", "numDeriv"))
```

**Julia (Case Study II).** The Julia scripts require Julia 1.x. Install the
required packages by running the following in the Julia REPL:

```julia
using Pkg
Pkg.add(["DifferentialEquations", "Turing", "Distributions", "DataFrames"])
```

> **Note:** verify exact dependencies against the `using` statements at the top
> of each script, as the lists above may be incomplete.

## Related resources

- [HazReg](https://github.com/FJRubio67/HazReg) — R package for parametric
  hazard-based regression models (overall and relative survival)
- [HazReg.jl](https://github.com/FJRubio67/HazReg.jl) — Julia implementation
  of the same
- [ODESurv](https://github.com/FJRubio67/ODESurv) — further R code for survival
  modelling via ODEs

## Citation

If you use this code, please cite:

```bibtex
@article{christen:2026,
  author  = {Christen, J.A. and Rubio, F.J.},
  title   = {Hazard-based distributional regression via ordinary differential equations},
  journal = {Statistical Methods in Medical Research},
  year    = {2026},
  note    = {in press},
  doi     = {10.1177/09622802251412840}
}
```
