# src/evaluation.py
"""
Evaluation utilities to compute quick metrics on a validation split.
Updated to be more robust to formatting, casing, and partial matches.
"""

import math
import re
import unicodedata
from typing import List, Dict, Tuple
import torch

def _normalize_text(text: str) -> str:
    """
    Agresiva normalización: minúsculas, quita acentos, quita puntuación no alfanumérica.
    Ayuda a que 'a)' coincida con 'A.'
    """
    if not text:
        return ""
    # 1. Lowercase
    text = text.lower().strip()
    # 2. Remove common prefixes
    text = re.sub(r"^(respuesta:|answer:|r:)\s*", "", text)
    # 3. Normalize unicode (quitar acentos)
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    # 4. Keep only alphanumeric and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # 5. Collapse spaces
    return " ".join(text.split())

def _compute_rouge_l_score(pred: str, target: str) -> float:
    """
    Calcula una aproximación de ROUGE-L (basada en la subsecuencia común más larga de tokens).
    Es mucho mejor para medir qué tanto de la frase original recuperó el modelo.
    """
    pred_tokens = _normalize_text(pred).split()
    target_tokens = _normalize_text(target).split()
    
    if not pred_tokens or not target_tokens:
        return 0.0
        
    m, n = len(pred_tokens), len(target_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == target_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    lcs_len = dp[m][n]
    
    # ROUGE-L Recall: Cuánto de la referencia recuperamos
    recall = lcs_len / n if n > 0 else 0
    # ROUGE-L Precision: Cuánto de lo generado es útil
    precision = lcs_len / m if m > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    # F1 Score balanceado
    f1 = 2 * ((precision * recall) / (precision + recall))
    return f1

def _masked_loss(model, input_ids, attention_mask, answer_len, question_len):
    """Compute loss only on the answer part to approximate PPL."""
    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, -100)
    if isinstance(question_len, (list, tuple)):
        for i, q_len in enumerate(question_len):
            labels[i, :q_len] = -100
    else:
        labels[:, :question_len] = -100
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss

def evaluate_model(
    model,
    tokenizer,
    val_data: List[Dict],
    device: str = "cpu", # Puede ser un objeto torch.device también
    max_samples: int = 16,
    gen_max_new_tokens: int = 64,
    eval_batch_size: int = 4,
) -> Tuple[float, float, float, float]:
    
    model.eval()
    
    # Manejo robusto del dispositivo
    if isinstance(device, str):
        device = torch.device(device)
    # No movemos el modelo aquí con .to() si es QLoRA, asumimos que ya está colocado.
    # Pero si el device es CPU, aseguramos.
    if device.type == 'cpu':
         model.to(device)

    acc_scores, rouge_scores, em_scores, losses = [], [], [], []
    
    samples = val_data[:max_samples]
    qa_texts = []
    prompt_lens = []

    with torch.no_grad():
        for ex in samples:
            question = ex.get("question", "")
            answer = ex.get("answer", "")
            context = ex.get("context", "")

            # Usamos el prompt mejorado (debe coincidir con el de training)
            prompt = (
                "Instrucciones: Eres un asistente experto en español. Tu tarea es extraer la respuesta EXACTA del contexto.\n"
                "1. Responde SIEMPRE en Español.\n"
                "2. Copia literalmente el texto del contexto que responde a la pregunta.\n"
                "3. No añadas introducciones ni traduzcas nada.\n\n"
                f"Contexto: {context}\n"
                f"Pregunta: {question}\n"
                "Respuesta:"
            )
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).to(device)

            gen_ids = model.generate(
                **inputs,
                max_new_tokens=gen_max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            prompt_len = inputs["input_ids"].shape[1]
            pred_full = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)
            
            # Post-procesado limpio
            pred_answer = pred_full.strip()
            # Quitamos comillas o intros iniciales
            pred_answer = pred_answer.lstrip('\n').lstrip('"').strip()
            # Si hay saltos de línea, cogemos la primera línea con contenido
            lines = [line.strip() for line in pred_answer.split('\n') if line.strip()]
            pred_answer = lines[0] if lines else ""

            # MÉTRICAS
            norm_pred = _normalize_text(pred_answer)
            norm_ref = _normalize_text(answer)

            # 1. Exact Match (Estricto)
            em_val = 1.0 if norm_pred == norm_ref and norm_ref != "" else 0.0
            
            # 2. Rouge-L (Suave/Semántico estructural)
            # Este será nuestro sustituto de F1 para texto
            rouge_val = _compute_rouge_l_score(pred_answer, answer)
            
            # 3. Accuracy "Soft" (Contención)
            # Si recuperamos más del 50% de los tokens correctos en orden, cuenta como acierto parcial.
            # O si la predicción está contenida en la referencia (corte)
            acc_val = 0.0
            if norm_ref in norm_pred and len(norm_ref) > 0:
                acc_val = 1.0
            elif norm_pred in norm_ref and len(norm_ref) > 0:
                # Penalizamos por ser corto, pero damos puntos
                acc_val = len(norm_pred) / len(norm_ref)
            else:
                # Si no es contención exacta, usamos el Rouge como proxy de accuracy
                acc_val = rouge_val if rouge_val > 0.3 else 0.0

            acc_scores.append(acc_val)
            rouge_scores.append(rouge_val)
            em_scores.append(em_val)
            
            qa_texts.append(prompt + " " + answer)
            prompt_lens.append(prompt_len)

        # Perplexity (PPL) calculation
        if qa_texts:
            for i in range(0, len(qa_texts), max(1, eval_batch_size)):
                batch_texts = qa_texts[i:i + max(1, eval_batch_size)]
                batch_q_lens = prompt_lens[i:i + max(1, eval_batch_size)]
                
                # Tokenizar batch
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

                if not torch.isfinite(loss):
                    loss = torch.tensor(20.0, device=device)
                losses.extend([loss.item()] * len(batch_texts))

    avg_acc = sum(acc_scores) / len(acc_scores) if acc_scores else 0.0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    
    avg_loss = sum(losses) / len(losses) if losses else float("inf")
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = 1e4
    if not math.isfinite(ppl):
        ppl = 1e4
    ppl = min(ppl, 1e4)

    # Devolvemos Rouge en lugar de F1 clásico, es mejor para frases largas
    return avg_acc, avg_rouge, avg_em, ppl