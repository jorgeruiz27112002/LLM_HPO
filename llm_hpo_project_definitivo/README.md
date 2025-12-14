# LLM Hyperparameter Optimization (HPO) Project

This repository contains the code and resources for the **LLM Hyperparameter Optimization** study, comparing **Genetic Algorithms (GA)** and **Random Search (RS)** for optimizing QLoRA fine-tuning of Large Language Models (specifically Gemma-2b).

## Repository Structure

- `src/`: Source code for the genetic algorithm, random search, dataset generation, and model evaluation.
- `run_hpo.py`: Main script to run the **Genetic Algorithm** optimization.
- `run_random_search.py`: Main script to run the **Random Search** baseline.
- `plot_fitness.py`: Utility to visualize fitness evolution from logs.
- `requirements.txt`: List of Python dependencies.
- `data/`: Directory for input PDFs and processed datasets.
- `logs/`: Directory where execution metrics and plots are saved.

## Setup and Installation

1.  **Clone the repository** (if applicable).
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 0. Data Preparation
Place your source PDF file (e.g., `source_book.pdf`) in `data/raw_pdf/`. The scripts will automatically generate the Q&A dataset from this PDF.

### 1. Running Genetic Algorithm (GA)
To run the HPO using Genetic Algorithms:

```bash
python run_hpo.py \
    --pdf-path data/raw_pdf/source_book.pdf \
    --pop-size 10 \
    --n-gen 5 \
    --train-steps 100 \
    --log-metrics --plot
```

**Common Arguments:**
- `--pop-size`: Size of the population (e.g., 10).
- `--n-gen`: Number of generations (e.g., 5).
- `--model-name`: Base model (default: `google/gemma-2b`).
- `--eval-samples`: Number of validation samples per evaluation.
- `--log-metrics`: Save metrics to `logs/ga_metrics.csv`.

### 2. Running Random Search (RS)
To run the Random Search baseline for comparison:

```bash
python run_random_search.py \
    --n-iter 20 \
    --train-steps 100 \
    --log-metrics --plot
```

**Key Arguments:**
- `--n-iter`: Number of random iterations to perform.

### 3. Visualization
If you ran with `--log-metrics`, you can generate fitness plots separately:

```bash
python plot_fitness.py
```
This looks for `logs/ga_metrics.csv` by default.

## Results
Process logs and metrics are saved in the `logs/` directory. 
- `ga_metrics.csv` tracks the fitness of individuals across generations in GA.
- `rs_metrics.csv` tracks the fitness of iterations in Random Search.
