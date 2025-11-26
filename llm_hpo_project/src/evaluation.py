"""
Evaluation utilities to compute quick metrics on a validation split.
Metrics: Accuracy (as EM), F1, Exact Match, Perplexity.
Designed to run on a small subset for fast GA feedback.
"""

import math
import re
import unicodedata
from typing import List, Dict, Tuple

import torch
from torch.nn.functional import cross_entropy


def _simple_tokenize(text: str) -> List[str]:
    return text.lower().strip().split()


def _f1_score(pred: str, target: str) -> float:
    pred_tokens = _simple_tokenize(pred)
    target_tokens = _simple_tokenize(target)
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0
    common = set(pred_tokens) & set(target_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def _strip_punctuation(text: str) -> str:
    # Remove punctuation and accents to reduce exact-match brittleness
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _normalize(text: str) -> str:
    text = text.lower().strip()
    # Remove common prefixes that the model might add
    text = re.sub(r"^(respuesta:|answer:)\s*", "", text)
    return _strip_punctuation(text)


def _exact_match(pred: str, target: str) -> float:
    return 1.0 if _normalize(target) == _normalize(pred) else 0.0


def _contains_match(pred: str, target: str) -> float:
    return 1.0 if _normalize(target) in _normalize(pred) else 0.0


def _masked_loss(model, input_ids, attention_mask, answer_len, question_len):
    """Compute loss only on the answer part to approximate PPL."""
    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, -100)
    if isinstance(question_len, (list, tuple)):
        for i, q_len in enumerate(question_len):
            labels[i, :q_len] = -100
    else:
        labels[:, :question_len] = -100  # ignore question tokens
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss


def evaluate_model(
    model,
    tokenizer,
    val_data: List[Dict],
    device: str = "cpu",
    max_samples: int = 16,
    gen_max_new_tokens: int = 32,
    eval_batch_size: int = 4,
) -> Tuple[float, float, float, float]:
    """
    Runs a light evaluation loop on a subset of validation data.

    Returns (Acc, F1, EM, PPL) averaged over samples.
    """

    model.eval()
    device = torch.device(device)
    model.to(device)

    accs, f1s, ems, losses = [], [], [], []

    samples = val_data[:max_samples]
    qa_texts = []
    prompt_lens = []

    with torch.no_grad():
        for ex in samples:
            question = ex.get("question", "")
            answer = ex.get("answer", "")
            context = ex.get("context", "")

            # Prompt con contexto y una instrucción clara de copia literal y respuesta concisa
            prompt = (
                "Responde usando SOLO texto copiado literalmente del contexto. No añadas nada más.\n"
                f"Context: {context}\nQuestion: {question}\nAnswer:"
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).to(device)

            # Generation for EM/F1
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=min(gen_max_new_tokens, 32),
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            # Keep only the generated part beyond the prompt length
            prompt_len = inputs["input_ids"].shape[1]
            pred_answer = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)
            # Keep only the first line to avoid trailing chatter
            pred_answer = pred_answer.split("\n")[0].strip()

            em_val = _exact_match(pred_answer, answer)
            acc_val = _contains_match(pred_answer, answer)
            f1_val = _f1_score(pred_answer, answer)

            accs.append(acc_val)
            f1s.append(f1_val)
            ems.append(em_val)
            qa_texts.append(prompt + " " + answer)
            prompt_lens.append(prompt_len)

        # Perplexity approximation in small batches
        for i in range(0, len(qa_texts), max(1, eval_batch_size)):
            batch_texts = qa_texts[i:i + max(1, eval_batch_size)]
            batch_q_lens = prompt_lens[i:i + max(1, eval_batch_size)]
            qa_ids = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).to(device)
            loss = _masked_loss(
                model,
                qa_ids["input_ids"],
                qa_ids.get("attention_mask"),
                answer_len=None,
                question_len=batch_q_lens,
            )

            # Evitar NaN/inf en la pérdida
            if not torch.isfinite(loss):
                loss = torch.tensor(20.0, device=device)  # pérdida grande pero finita pero menos extrema
            losses.extend([loss.item()] * len(batch_texts))

    avg_acc = sum(accs) / len(accs) if accs else 0.0
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    avg_em = sum(ems) / len(ems) if ems else 0.0
    avg_loss = sum(losses) / len(losses) if losses else float("inf")
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")
    if not math.isfinite(ppl):
        ppl = float("inf")
    ppl = min(ppl, 1e4)  # tope más bajo para evitar saturar siempre

    return avg_acc, avg_f1, avg_em, ppl
