# src/genetic_algorithm.py
"""
Genetic Algorithm for hyperparameter optimization of the LLM.

- Mixed search space: continuous + discrete (defined in search_space.py)
- Custom mutation handling both types
- Uses a fitness function from fitness.py
- Uses multiprocessing to evaluate individuals in parallel on multiple GPUs.
"""

import math
import random
import copy
import csv
import os
import torch
import torch.multiprocessing as mp
from deap import base, creator, tools, algorithms

from .search_space import search_space
from .fitness import compute_fitness
from .model_wrapper import load_model
from .dataset_generator import load_validation_data, load_train_data
from .evaluation import evaluate_model


def _is_oom_error(err: RuntimeError) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "mps backend out of memory" in msg


def _clear_mps_cache():
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    # answer = example.get("answer", "") # Solo si la usas en el prompt (few-shot), si no, quítala.
    
    # CAMBIO: Instrucciones mucho más explícitas
    return (
        "Instrucciones: Eres un asistente experto en español. Tu tarea es extraer la respuesta EXACTA del contexto.\n"
        "1. Responde SIEMPRE en Español.\n"
        "2. Copia literalmente el texto del contexto que responde a la pregunta.\n"
        "3. No añadas introducciones ni traduzcas nada.\n\n"
        f"Contexto: {context}\n"
        f"Pregunta: {question}\n"
        "Respuesta:"
    )


def _write_log(buffer, path):
    if buffer is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(buffer[0].keys()) if buffer else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(buffer)


def run_proxy_training(hparams, device="cuda", base_model=None, max_eval_samples=4, max_train_steps=150, cached=None):
    """
    Entrenamiento breve + evaluación para el GA.
    """
    
    # --- CORRECCIÓN CRÍTICA: Eliminamos el sistema de caché local ---
    # No podemos usar load_state_dict con QLoRA/bitsandbytes fácilmente,
    # y además los hparams (arquitectura) cambian entre individuos.
    # Confiamos en model_wrapper para el caché del modelo BASE.
    model, tokenizer = load_model(hparams, device=device, base_model=base_model)
    
    train_data = load_train_data()
    val_data = load_validation_data()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["eta"], weight_decay=hparams["weight_decay"])

    # Detectar dónde cargó realmente el modelo para enviar los datos allí
    if hasattr(model, "device"):
        device_t = model.device
    else:
        try:
            device_t = next(model.parameters()).device
        except StopIteration:
            device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fallback si reporta CPU pero esperamos GPU
    if device_t.type == 'cpu' and device and "cuda" in str(device):
        device_t = torch.device(device)

    epochs = max(1, int(hparams.get("epochs", 1)))
    batch_size = max(1, int(hparams["batch_size"]))
    
    # Gradient Accumulation Logic
    max_phys_batch = 8 
    phys_batch = max(1, min(batch_size, max_phys_batch))
    accum_steps = max(1, math.ceil(batch_size / phys_batch))
    
    effective_seq_len = max(64, int(hparams["seq_len"]))
    dataset_size = len(train_data)
    steps_per_epoch = math.ceil(dataset_size / batch_size) if dataset_size else 0
    effective_max_steps = max(
        int(max_train_steps) if max_train_steps is not None else 0,
        steps_per_epoch * epochs,
    )

    step_count = 0
    for _ in range(epochs):
        if dataset_size == 0:
            break
        indices = list(range(dataset_size))
        random.shuffle(indices)
        pos = 0
        accum_counter = 0
        while pos < dataset_size and step_count < effective_max_steps:
            cur_bs = min(phys_batch, dataset_size - pos)
            batch_idx = indices[pos:pos + cur_bs]
            batch = [train_data[i] for i in batch_idx]
            texts = [_build_prompt(ex) for ex in batch]
            try:
                tokens = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=effective_seq_len,
                ).to(device_t) # Movemos los datos AL MISMO sitio que el modelo
                
                labels = tokens["input_ids"].clone()
                outputs = model(**tokens, labels=labels)
                loss = outputs.loss / accum_steps
                loss.backward()
                accum_counter += 1
                
                if accum_counter % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    step_count += 1
                pos += cur_bs
            except RuntimeError as e:
                if _is_oom_error(e):
                    _clear_mps_cache()
                    if phys_batch > 1:
                        phys_batch = max(1, phys_batch // 2)
                        accum_steps = max(1, math.ceil(batch_size / phys_batch))
                        print(f"[GA][OOM] Reducing physical batch to {phys_batch} (accum {accum_steps}) and retrying...")
                        continue
                    if effective_seq_len > 256:
                        effective_seq_len = max(256, effective_seq_len // 2)
                        print(f"[GA][OOM] Reducing seq_len to {effective_seq_len} and retrying...")
                        continue
                raise
        if step_count >= effective_max_steps:
            break

    model.eval()

    Acc, F1, EM, PPL = evaluate_model(
        model,
        tokenizer,
        val_data,
        device=device_t, 
        max_samples=max_eval_samples,
        gen_max_new_tokens=64, 
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


def evaluate_individual(individual, device="cuda", base_model=None, max_eval_samples=4, cached=None, max_train_steps=150):
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
# PARALLEL WORKER
# =========================
def worker_eval(args):
    """
    Função que roda dentro de cada processo separado.
    Recebe: (individuo_raw, gpu_id, model_name, max_eval_samples, max_train_steps)
    Retorna: (fitness_tuple, metrics)
    """
    ind_data, gpu_id, model_name, max_eval, max_steps = args
    
    # Configura GPU específica para este proceso
    device_str = f"cuda:{gpu_id}"
  
    class DummyInd(list):
        pass
    
    ind_obj = DummyInd(ind_data)
    
    try:
        # Importante: cached=None para forzar carga nueva en este proceso/GPU
        fitness = evaluate_individual(
            ind_obj,
            device=device_str,
            base_model=model_name,
            max_eval_samples=max_eval,
            cached=None, 
            max_train_steps=max_steps
        )
        metrics = getattr(ind_obj, "metrics", None)
        return fitness, metrics
    except Exception as e:
        print(f"[Worker Error GPU {gpu_id}] {e}")
        # Retorna fitness ruim en caso de error
        return (-9999.0,), None


def parallel_evaluate_population(population, pool, num_gpus, model_name, max_eval, max_steps):
    """
    Helper para distribuir a população entre as GPUs
    """
    tasks = []
    for i, ind in enumerate(population):
        # Round Robin: Indivíduo 0 -> GPU 0, Ind 1 -> GPU 1... Ind 8 -> GPU 0
        gpu_id = i % num_gpus 
        # ind[:] para enviar apenas los datos, no el objeto complejo del DEAP
        tasks.append((ind[:], gpu_id, model_name, max_eval, max_steps))
    
    results = pool.map(worker_eval, tasks)
    
    # Atribuir resultados de vuelta a la población original
    for ind, (fit, met) in zip(population, results):
        ind.fitness.values = fit
        if met is not None:
            ind.metrics = met


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
    max_train_steps=150,
    log_metrics=False,
    log_path=None,
    lambda_size=None,
):
    # Configuración de multiprocessamento para CUDA (Obligatorio 'spawn')
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # DETECCIÓN Y SANITIZACIÓN DE DISPOSITIVO
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Detectar GPUs reales para el paralelismo
    if device == "cuda" and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"[GA] Modo Paralelo Ativado: Usando {num_gpus} GPUs.")
    else:
        num_gpus = 0
        print(f"[GA] Aviso: Ejecutando en modo serial en {device}.")
        
    pop_size = pop_size or POP_SIZE
    n_gen = n_gen or N_GEN
    lambda_size = lambda_size or pop_size
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

    # Estadísticas simples
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda fits: sum(fits) / len(fits))
    stats.register("max", max)
    stats.register("min", min)

    print("[GA] Starting evolution...")

    # ==========================
    # EVALUACIÓN GENERACIÓN 0
    # ==========================
    if num_gpus > 1:
        with mp.Pool(processes=num_gpus) as pool:
            print(f"[GA] Gen 0 - Evaluando {len(population)} indivíduos en paralelo...")
            parallel_evaluate_population(
                population, pool, num_gpus, model_name, max_eval_samples, max_train_steps
            )
    else:
        # Fallback para o modo antigo (serial) se tiver só 1 GPU ou CPU
        for idx, ind in enumerate(population, 1):
            print(f"[GA] Gen 0 - evaluando individuo {idx}/{len(population)}")
            ind.fitness.values = evaluate_individual(
                ind,
                device=device,
                base_model=model_name,
                max_eval_samples=max_eval_samples,
                cached=cache,
                max_train_steps=max_train_steps,
            )

    population = _ensure_unique_population(
        population,
        pop_size,
        device=device,
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
        elite_ids = set(map(id, tools.selBest(population, ELITE_K))) if ELITE_K else set()
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
                "is_elite": 1 if id(ind) in elite_ids else 0,
                **ind_h,
            })
        _write_log(log_buffer, log_path)

    # ==========================
    # BUCLE EVOLUTIVO
    # ==========================
    for gen in range(1, n_gen + 1):
        elites = tools.selBest(population, ELITE_K) if ELITE_K else []
        elites = list(map(toolbox.clone, elites))

        diversity = _population_diversity(population)
        cx_prob = _adapt_cx_prob(diversity)
        mut_prob = _adapt_mut_prob(diversity)

        offspring = toolbox.select(population, lambda_size)
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                if hasattr(child1, "metrics"):
                    del child1.metrics
                if hasattr(child2, "metrics"):
                    del child2.metrics

        # Mutación
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                if hasattr(mutant, "metrics"):
                    del mutant.metrics

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        # Evaluación Paralela de Hijos
        if num_gpus > 1:
            if invalid_ind:
                print(f"[GA] Gen {gen} - Evaluando {len(invalid_ind)} hijos en paralelo...")
                with mp.Pool(processes=num_gpus) as pool:
                    parallel_evaluate_population(
                        invalid_ind, pool, num_gpus, model_name, max_eval_samples, max_train_steps
                    )
        else:
            # Modo Serial
            for idx, ind in enumerate(invalid_ind, 1):
                print(f"[GA] Gen {gen} - evaluando hijo {idx}/{len(invalid_ind)}")
                ind.fitness.values = evaluate_individual(
                    ind,
                    device=device,
                    base_model=model_name,
                    max_eval_samples=max_eval_samples,
                    cached=cache,
                    max_train_steps=max_train_steps,
                )
                
        # Selección (μ+λ)
        combined = population + offspring
        survivors = tools.selBest(combined, max(pop_size - ELITE_K, 0))
        population = elites + survivors
        
        # Relleno de población (Unique Check)
        population = _ensure_unique_population(
            population,
            pop_size,
            device=device,
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
            elite_ids = set(map(id, elites)) if ELITE_K else set()
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
                    "is_elite": 1 if id(ind) in elite_ids else 0,
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