# genetic_algorithm.py
"""
Genetic Algorithm for hyperparameter optimization of the LLM.

- Mixed search space: continuous + discrete (defined in search_space.py)
- Custom mutation handling both types
- Uses a fitness function from fitness.py
- Por ahora, usa métricas DUMMY para que el GA se pueda ejecutar.
  Más adelante sustituiremos run_proxy_training() por el entrenamiento real.
"""

import random
import copy
import csv
import os
import torch
from deap import base, creator, tools, algorithms

from .search_space import search_space
from .fitness import compute_fitness
from .model_wrapper import load_model
from .dataset_generator import load_validation_data, load_train_data
from .evaluation import evaluate_model


# =========================
# GA CONFIG
# =========================
POP_SIZE = 10
N_GEN = 5
CX_PB = 0.9
MUT_PB = 0.8
INDPB = 0.2      # prob de mutar cada gen
TOURNAMENT_K = 3
ELITE_K = 1      # número de élites que se preservan cada generación (copiados intactos)
ETA_C = 15       # parámetro de SBX (crossover)
ETA_M = 20       # parámetro de mutación polinómica


# =========================
# INDIVIDUAL ENCODING
# =========================
param_names = list(search_space.keys())


def create_individual():
    """
    Crea un individuo respetando el tipo de cada hiperparámetro.
    - Tuplas => continuo (uniforme en el rango)
    - Listas => discreto (elige un valor de la lista)
    """
    genes = []
    for key in param_names:
        space = search_space[key]
        if isinstance(space, tuple):  # continuo
            low, high = space
            genes.append(random.uniform(low, high))
        else:  # discreto
            genes.append(random.choice(space))
    return genes


def decode_individual(individual):
    """
    Convierte la lista de genes en un diccionario {nombre_param: valor}
    """
    return {name: val for name, val in zip(param_names, individual)}


# =========================
# SBX + MUTACIÓN POLINÓMICA
# =========================
def _sbx_crossover_gene(x1, x2, low, high, eta=ETA_C):
    if abs(x1 - x2) < 1e-12:
        return x1, x2
    # Aseguramos orden
    if x1 > x2:
        x1, x2 = x2, x1
    rand = random.random()
    beta = 1.0 + (2.0 * (x1 - low) / (x2 - x1))
    alpha = 2.0 - beta ** -(eta + 1.0)
    if rand <= 1.0 / alpha:
        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
    else:
        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
    c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

    beta = 1.0 + (2.0 * (high - x2) / (x2 - x1))
    alpha = 2.0 - beta ** -(eta + 1.0)
    if rand <= 1.0 / alpha:
        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
    else:
        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
    c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

    c1 = min(max(c1, low), high)
    c2 = min(max(c2, low), high)
    return c1, c2


def sbx_crossover(ind1, ind2, eta=ETA_C):
    """
    Simulated Binary Crossover (SBX) para genes continuos.
    Para genes discretos se hace intercambio uniforme.
    """
    for i, key in enumerate(param_names):
        space = search_space[key]
        if isinstance(space, tuple):
            low, high = space
            ind1[i], ind2[i] = _sbx_crossover_gene(ind1[i], ind2[i], low, high, eta=eta)
        else:
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


def mutate_individual(individual, indpb=INDPB, eta=ETA_M):
    """
    Mutación polinómica para genes continuos y cambio discreto para categóricos.
    """
    for i, key in enumerate(param_names):
        if random.random() > indpb:
            continue

        space = search_space[key]

        # continuo: mutación polinómica
        if isinstance(space, tuple):
            low, high = space
            x = individual[i]
            if high == low:
                continue
            delta1 = (x - low) / (high - low)
            delta2 = (high - x) / (high - low)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.0)
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                deltaq = 1.0 - val ** mut_pow
            x = x + deltaq * (high - low)
            x = min(max(x, low), high)
            individual[i] = x

        # discreto: cambiar a otro valor posible
        else:
            current = individual[i]
            choices = [c for c in space if c != current]
            if not choices:
                continue
            individual[i] = random.choice(choices)

    return individual,


# =========================
def _build_prompt(example):
    context = example.get("context", "")
    question = example.get("question", "")
    answer = example.get("answer", "")
    prompt = (
        "Responde usando SOLO texto copiado literalmente del contexto. No añadas nada más.\n"
        f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
    )
    return prompt


def _write_log(buffer, path):
    if buffer is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(buffer[0].keys()) if buffer else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(buffer)


def run_proxy_training(hparams, device="mps", base_model=None, max_eval_samples=4, max_train_steps=20, cached=None):
    """
    Entrenamiento breve + evaluación para el GA.
    """

    if cached and {"model", "tokenizer", "init_state"} <= set(cached.keys()):
        model = cached["model"]
        model.load_state_dict(cached["init_state"])
        tokenizer = cached["tokenizer"]
    else:
        model, tokenizer = load_model(hparams, device=device, base_model=base_model)
        if cached is not None:
            cached["model"] = model
            cached["tokenizer"] = tokenizer
            cached["init_state"] = copy.deepcopy(model.state_dict())
    train_data = load_train_data()
    val_data = load_validation_data()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["eta"], weight_decay=hparams["weight_decay"])

    device_t = torch.device(device)
    model.to(device_t)

    epochs = max(1, int(hparams.get("epochs", 1)))
    steps_per_epoch = max(1, int(max_train_steps))

    for _ in range(epochs):
        random.shuffle(train_data)
        for _ in range(steps_per_epoch):
            batch = random.sample(train_data, min(hparams["batch_size"], len(train_data)))
            texts = [_build_prompt(ex) for ex in batch]
            tokens = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=hparams["seq_len"],
            ).to(device_t)
            labels = tokens["input_ids"].clone()
            outputs = model(**tokens, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    model.eval()

    Acc, F1, EM, PPL = evaluate_model(
        model,
        tokenizer,
        val_data,
        device=device,
        max_samples=max_eval_samples,
        gen_max_new_tokens=32,
        eval_batch_size=max(1, min(4, max_eval_samples)),
    )

    max_bs = max(search_space["batch_size"])
    max_seq = max(search_space["seq_len"])
    max_ep = max(search_space["epochs"])

    C = (
        hparams["batch_size"] / max_bs +
        hparams["seq_len"] / max_seq +
        hparams["epochs"] / max_ep
    ) / 3.0  # normalizado a [0,1]

    return Acc, F1, EM, PPL, C


def evaluate_individual(individual, device="mps", base_model=None, max_eval_samples=4, cached=None, max_train_steps=20):
    """
    Evalúa un individuo con entrenamiento corto + métricas.
    """
    h = decode_individual(individual)
    metrics = run_proxy_training(
        h,
        device=device,
        base_model=base_model,
        max_eval_samples=max_eval_samples,
        max_train_steps=max_train_steps,
        cached=cached,
    )
    fitness_value = compute_fitness(metrics)
    individual.metrics = metrics  # guardar métricas para logging posterior
    return (fitness_value,)


# =========================
# DIVERSIDAD Y ADAPTACIÓN
# =========================
def _population_diversity(population):
    """
    Estima diversidad en [0,1] promediando diversidad normalizada por gen.
    - Continuos: (max - min) / rango
    - Discretos: fracción de valores únicos
    """
    if not population:
        return 0.0
    divs = []
    for idx, key in enumerate(param_names):
        space = search_space[key]
        values = [ind[idx] for ind in population]
        if isinstance(space, tuple):
            low, high = space
            if high == low:
                divs.append(0.0)
            else:
                spread = max(values) - min(values)
                divs.append(min(max(spread / (high - low), 0.0), 1.0))
        else:
            unique_frac = len(set(values)) / len(values)
            divs.append(unique_frac)
    return sum(divs) / len(divs)


def _adapt_cx_prob(diversity):
    """
    Aumenta cruce con mayor diversidad; evita caer por debajo de 0.2 ni exceder 0.95.
    """
    return min(0.95, max(0.2, CX_PB * (0.5 + 0.5 * diversity)))


def _adapt_mut_prob(diversity):
    """
    Aumenta mutación cuando la diversidad cae; acota entre 0.05 y 0.95.
    """
    factor = 1.5 - diversity  # +50% si diversidad es 0, reduce si es alta
    return min(0.95, max(0.05, MUT_PB * factor))


def _ensure_unique_population(
    population,
    pop_size,
    device,
    base_model,
    max_eval_samples,
    max_train_steps,
    cache,
    gen_label="?",
):
    """
    Filtra duplicados manteniendo los mejores y rellena con nuevos individuos si falta tamaño.
    """
    unique = []
    seen = set()
    sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
    for ind in sorted_pop:
        key = tuple(ind)
        if key in seen:
            continue
        unique.append(ind)
        seen.add(key)
        if len(unique) >= pop_size:
            break

    attempts = 0
    while len(unique) < pop_size and attempts < pop_size * 3:
        attempts += 1
        new_ind = creator.Individual(create_individual())
        print(f"[GA] Gen {gen_label} - rellenando individuo {len(unique)+1}/{pop_size}")
        new_ind.fitness.values = evaluate_individual(
            new_ind,
            device=device,
            base_model=base_model,
            max_eval_samples=max_eval_samples,
            cached=cache,
            max_train_steps=max_train_steps,
        )
        key = tuple(new_ind)
        if key in seen:
            continue
        unique.append(new_ind)
        seen.add(key)

    return unique


# =========================
# GA MAIN
# =========================
def run_ga(
    pop_size=POP_SIZE,
    n_gen=N_GEN,
    seed=42,
    model_name=None,
    device=None,
    max_eval_samples=8,
    max_train_steps=30,
    log_metrics=False,
    log_path=None,
    lambda_size=None,
):
    pop_size = pop_size or POP_SIZE
    n_gen = n_gen or N_GEN
    lambda_size = lambda_size or pop_size  # λ puede diferir de µ, por defecto igual
    if ELITE_K > pop_size:
        raise ValueError("ELITE_K no puede ser mayor que pop_size")
    random.seed(seed)

    # Crear clases de DEAP si no existen ya
    try:
        creator.FitnessMax
    except AttributeError:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    try:
        creator.Individual
    except AttributeError:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    cache = {}
    log_buffer = [] if log_metrics else None

    toolbox = base.Toolbox()
    toolbox.register("mate", sbx_crossover, eta=ETA_C)
    toolbox.register("mutate", mutate_individual, indpb=INDPB, eta=ETA_M)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_K)

    # Población inicial
    population = [creator.Individual(create_individual()) for _ in range(pop_size)]

    # Hall of fame (mejores individuos)
    hof = tools.HallOfFame(1)

    # Estadísticas simples (solo fitness medio, máx, mín)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda fits: sum(fits) / len(fits))
    stats.register("max", max)
    stats.register("min", min)

    print("[GA] Starting evolution...")

    # Evaluación inicial (gen 0)
    for idx, ind in enumerate(population, 1):
        print(f"[GA] Gen 0 - evaluando individuo {idx}/{len(population)}")
        ind.fitness.values = evaluate_individual(
            ind,
            device=device or "mps",
            base_model=model_name,
            max_eval_samples=max_eval_samples,
            cached=cache,
            max_train_steps=max_train_steps,
        )

    population = _ensure_unique_population(
        population,
        pop_size,
        device=device or "mps",
        base_model=model_name,
        max_eval_samples=max_eval_samples,
        max_train_steps=max_train_steps,
        cache=cache,
        gen_label="0",
    )

    logbook = tools.Logbook()
    record = stats.compile(population)
    logbook.record(gen=0, nevals=len(population), **record)
    if log_metrics and log_path:
        for ind in population:
            ind_h = decode_individual(ind)
            if hasattr(ind, "metrics"):
                Acc, F1, EM, PPL, C = ind.metrics
            else:
                Acc = F1 = EM = PPL = C = ""
            log_buffer.append({
                "generation": 0,
                "fitness": ind.fitness.values[0],
                "Acc": Acc,
                "F1": F1,
                "EM": EM,
                "PPL": PPL,
                "C": C,
                "is_elite": 1,
                **ind_h,
            })
        _write_log(log_buffer, log_path)

    # Evolución (elitismo explícito + μ+λ): copiamos élites intactos y el resto se decide con padres+hijos
    for gen in range(1, n_gen + 1):
        elites = tools.selBest(population, ELITE_K) if ELITE_K else []
        elites = list(map(toolbox.clone, elites))

        diversity = _population_diversity(population)
        cx_prob = _adapt_cx_prob(diversity)
        mut_prob = _adapt_mut_prob(diversity)

        offspring = toolbox.select(population, lambda_size)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                if hasattr(child1, "metrics"):
                    del child1.metrics
                if hasattr(child2, "metrics"):
                    del child2.metrics

        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                if hasattr(mutant, "metrics"):
                    del mutant.metrics

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for idx, ind in enumerate(invalid_ind, 1):
            print(f"[GA] Gen {gen} - evaluando hijo {idx}/{len(invalid_ind)}")
            ind.fitness.values = evaluate_individual(
                ind,
                device=device or "mps",
                base_model=model_name,
                max_eval_samples=max_eval_samples,
                cached=cache,
                max_train_steps=max_train_steps,
            )

        # Selección (μ+λ): mejores de padres + hijos; los élites ya están reservados
        combined = population + offspring
        survivors = tools.selBest(combined, max(pop_size - ELITE_K, 0))
        population = elites + survivors
        population = _ensure_unique_population(
            population,
            pop_size,
            device=device or "mps",
            base_model=model_name,
            max_eval_samples=max_eval_samples,
            max_train_steps=max_train_steps,
            cache=cache,
            gen_label=str(gen),
        )
        hof.update(population)

        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if log_metrics and log_path:
            # Loguear todos los individuos seleccionados tras la supervivencia de la generación
            for ind in population:
                ind_h = decode_individual(ind)
                if hasattr(ind, "metrics"):
                    Acc, F1, EM, PPL, C = ind.metrics
                else:
                    Acc = F1 = EM = PPL = C = ""
                log_buffer.append({
                    "generation": gen,
                    "fitness": ind.fitness.values[0],
                    "Acc": Acc,
                    "F1": F1,
                    "EM": EM,
                    "PPL": PPL,
                    "C": C,
                    "is_elite": 1,  # marcamos como mejores de la generación
                    **ind_h,
                })
            _write_log(log_buffer, log_path)

    best_ind = hof[0]
    best_hparams = decode_individual(best_ind)
    best_fitness = best_ind.fitness.values[0]

    print("\n[GA] Best individual:", best_hparams)
    print("[GA] Best fitness:", best_fitness)

    return best_hparams, best_fitness, logbook


# Modo script
if __name__ == "__main__":
    run_ga()
