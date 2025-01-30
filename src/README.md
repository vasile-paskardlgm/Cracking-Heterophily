## source code for example implementation of CTH .

- **Important note** The datasets used in our paper—Gamers, Pokec, and Wiki—are sourced from this [paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/ae816a80e4c1c56caa2eb4e1819cbb2f-Abstract.html), with the corresponding GitHub repository available in [this link](https://github.com/CUAI/Non-Homophily-Large-Scale). However, we have recently observed that access to these datasets, which are hosted on Google Drive, now requires permission from the repository owner. To comply with copyright and licensing terms, please contact the repository owner to obtain access and update the dataset URL in `dataset.py` as needed.
- Please ensure that your GPU has adequate memory. Our experiments were conducted using two NVIDIA A100 GPUs, each with 80GB of memory. If your system does not meet this requirement, you may utilize the CPU mode we provide as an alternative (see the *Usage* section in the following).
- The demo code currently provides *GCN* as the backbone. However, `network.py` is designed for easy customization, allowing you to incorporate any GNN of your choice.

### Usage

CTH consists of two key steps: *1) Similarity-guided coarsening* and *2) Residual-reintegrated training*, which are implemented in two Python files:

- `preprocessing.py`: Generates augmented features and the coarsened graph, which would be stored in the `preprocessed` folder.
- `main.py`: Performs GNN training using the CTH method.

Therefore, to run CTH on the *Gamers* dataset with a *GCN* backbone and a coarsening ratio of *0.1*, execute:

> python preprocessing.py --dataset twitch-gamer --M 5 --coarsening_ratio 0.1 --gpu
> python main.py --dataset twitch-gamer --epochs 500 --early_stopping 50 --coarsening_ratio 0.1 --hidden 128 --num_layers 3 --dropout 0.5 --gpu

If GPU memory is insufficient, simply remove `--gpu` to run on CPU.