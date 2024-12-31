# Diffusion Module

This directory contains the core implementation for training and evaluating diffusion models. It includes tools for dataset handling, training, and evaluation.

---

## Structure

```
diffusion/
│
├── datasets/               # Dataset loaders
│   ├── base_dataset.py     # Abstract base class for datasets
│   ├── simulated_dataset.py# Simulated dataset with geometric shapes
│   ├── nuscenes_dataset.py # Placeholder for nuScenes dataset
│   └── __init__.py
│
├── modules/                # Model definitions
│   ├── diffusion_model.py  # Custom diffusion model implementation
│   └── __init__.py
│
├── training/               # Training loop
│   ├── trainer.py          # Training script using PyTorch Lightning
│   └── __init__.py
│
├── evaluation/             # Evaluation tools
│   ├── eval_tool.py        # Streamlit-based evaluation tool
│   └── __init__.py
│
└── __init__.py             # Marks the `diffusion` directory as a package
```

---

## Requirements

Ensure the following are installed in your environment:
- Python 3.10
- PyTorch
- PyTorch Lightning
- Streamlit

You can install these dependencies using the `requirements.txt` in the project root.

---

## Usage

### Adding the Root Directory to PYTHONPATH

To run any script in this directory, ensure the root of the repository is added to `PYTHONPATH`. This allows Python to resolve the imports correctly.

1. Temporarily add the root directory:
    ```bash
    export PYTHONPATH=$(pwd)
    ```

2. Make it persistent by adding this to your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`):
    ```bash
    echo 'export PYTHONPATH=$(pwd)' >> ~/.bashrc
    source ~/.bashrc
    ```

---

## Running Training

To start training a diffusion model:
```bash
python src/diffusion/training/trainer.py
```

This script trains the diffusion model using PyTorch Lightning.

---

## Running Evaluation

To evaluate the diffusion model using a Streamlit-based evaluation tool:
```bash
streamlit run src/diffusion/evaluation/eval_tool.py
```

- Open the Streamlit app in your browser at the URL provided.
- Interact with the app to visualize and analyze the diffusion model's output.

---

## Contributing

Feel free to add new datasets, improve models, or extend the evaluation tool. Submit pull requests to propose changes.

---

## License

This project is licensed under the MIT License. See the LICENSE file in the project root for details.