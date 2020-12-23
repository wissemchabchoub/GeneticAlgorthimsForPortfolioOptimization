<h3 align="center">Genetic Algorithms for Portfolio Optimization</h3>

---

[Wissem Chabchoub](https://www.linkedin.com/in/wissem-chabchoub/) | [Contact us](mailto:chb.wissem@gmail.com)

## 📝 Table of Contents

- [📝 Table of Contents](#-table-of-contents)
- [🧐 About <a name = "about"></a>](#-about)
- [🎥 Repository Structure  <a name = "repo-struct"></a>](#-repository-structure)


## 🧐 About <a name = "about"></a>

In this project, we are using NSGA II algorithm to optimize a constrained portfolio. These constraints include no short selling, cardinality, and quantity constraints.

<p align="center">
  <img src="pareto.png?raw=true" />
</p>



## 🎥 Repository Structure  <a name = "repo-struct"></a>


1. `data`: Where the input prices are saved and the optimized portfolios will be saved
2. `nsga2`: Contains the adapted nsga2 python implementation
3. `Optimization.py`: The main optimization class
4. `Test.ipynb`: a Jupyter notebook for running and testing
