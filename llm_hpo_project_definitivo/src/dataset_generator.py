# dataset_generator.py
"""
Dataset generation module for the LLM hyperparameter optimization project.

Pipeline:
1. Extract text from PDF.
2. Generate extractive and closed-ended Q&A pairs.
3. Filter noisy / redundant questions using Sentence-BERT embeddings.
4. Split into train/validation/test.
"""
import torch
import os
import json
import random
import re
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split


# Lazy load to avoid paying the cost if dataset is already cached
_bert = None


def _get_bert():
    global _bert
    if _bert is None:
        print("[Dataset] Loading Sentence-BERT model...")
        _bert = SentenceTransformer("all-MiniLM-L6-v2")
    return _bert


# ---------------------------
# Step 1: Extract PDF text
# ---------------------------
def extract_pdf_text(pdf_path):
    print(f"[Dataset] Extracting text from: {pdf_path}")
    raw = extract_text(pdf_path)
    return _clean_text(raw)


def _clean_text(text: str) -> str:
    """
    Limpia artefactos comunes de PDF:
    - Une palabras cortadas por guiones al final de línea.
    - Colapsa saltos de línea múltiples en espacios.
    - Colapsa espacios repetidos.
    """
    if not text:
        return ""
    # une palabras cortadas por guiones de fin de línea
    text = re.sub(r"-\s*\n\s*", "", text)
    # reemplaza saltos de línea por espacios
    text = text.replace("\r", " ")
    text = re.sub(r"\n+", " ", text)
    # colapsa espacios extra
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _split_sentences(text):
    # Divide por delimitadores fuertes y limpia espacios
    raw_parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for part in raw_parts:
        p = part.strip()
        if len(p) < 40 or len(p) > 400:
            continue
        # descarta líneas con muchos dígitos o formatos raros (tablas)
        if sum(ch.isdigit() for ch in p) > len(p) * 0.25:
            continue
        sentences.append(p)
    return sentences


def _language_templates(language: str):
    """
    Small helper to keep preguntas/respuestas alineadas con el idioma del PDF.
    Actualmente soporta español/inglés; por defecto español para datasets en ES.
    """
    lang = (language or "es").lower()
    if lang.startswith("es"):
        return {
            "extractive": 'Según el contexto, ¿qué explica esta frase: "{snippet}"?',
            "closed": '¿Cuál es la {topic} de la siguiente afirmación? "{snippet}"?',
            "topics": [
                "definición", "propósito", "ventajas", "características",
                "limitaciones", "pasos", "componentes", "objetivo", "impacto", "requisitos",
            ],
        }
    return {
        "extractive": 'According to the context, what does this sentence explain: "{snippet}"?',
        "closed": 'What is the {topic} of the following statement? "{snippet}"?',
        "topics": [
            "definition", "purpose", "advantages", "characteristics",
            "limitations", "steps", "components", "goal", "impact", "requirements",
        ],
    }


# ---------------------------
# Step 2: Create synthetic Q&A
# ---------------------------
def build_extractive_qa(text, language="es"):
    """
    Extractive Q&A:
    - Split by sentences
    - Turn each into a Q&A pair
    """
    print("[Dataset] Building extractive Q&A...")
    sentences = _split_sentences(text)
    qa_pairs = []
    seen_contexts = set()
    templates = _language_templates(language)

    for s in sentences:
        if s in seen_contexts:
            continue
        seen_contexts.add(s)

        snippet = s[:120] + ("..." if len(s) > 120 else "")
        q = templates["extractive"].format(snippet=snippet)
        qa_pairs.append({
            "type": "extractive",
            "question": q,
            "answer": s,
            "context": s  # usamos la frase original como contexto
        })

    return qa_pairs


def build_closed_ended_qa(text, language="es"):
    """
    Closed-ended Q&A:
    - Generate short definition-like questions
    """
    print("[Dataset] Building closed-ended Q&A...")
    questions = []
    templates = _language_templates(language)
    topics = templates["topics"]
    closed_tpl = templates["closed"]

    sentences = _split_sentences(text)
    seen_pairs = set()
    for s in sentences:
        if len(s) < 80:
            continue

        topic = random.choice(topics)
        snippet = s[:110] + ("..." if len(s) > 110 else "")
        q = closed_tpl.format(topic=topic, snippet=snippet)
        if (topic, snippet) in seen_pairs:
            continue
        seen_pairs.add((topic, snippet))

        questions.append({
            "type": "closed",
            "question": q,
            "answer": s,
            "context": s  # contexto mínimo: la misma frase de origen
        })

    return questions


# ---------------------------
# Step 3: Semantic Filtering
# ---------------------------
def semantic_filtering(qa_pairs, threshold=0.75):
    """
    Remove near-duplicate questions using embeddings.
    """
    print("[Dataset] Semantic filtering with threshold =", threshold)

    filtered = []
    embeddings = None  # tensor acumulado (N, D)

    encoder = _get_bert()

    for entry in qa_pairs:
        q = entry["question"]
        emb = encoder.encode(q, convert_to_tensor=True)

        if embeddings is not None:
            sims = util.cos_sim(emb.unsqueeze(0), embeddings)  # (1, N)
            max_sim = sims.max().item()
            if max_sim > threshold:
                continue

        emb_batch = emb.unsqueeze(0)
        embeddings = emb_batch if embeddings is None else torch.cat((embeddings, emb_batch), dim=0)
        filtered.append(entry)

    print(f"[Dataset] Filtered down to {len(filtered)} samples.")
    return filtered


# ---------------------------
# Step 4: Save splits
# ---------------------------
def split_and_save(qa_pairs, output_dir, language="es", pdf_path=None):
    print("[Dataset] Splitting dataset...")

    train, temp = train_test_split(qa_pairs, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    os.makedirs(output_dir, exist_ok=True)

    json.dump(train, open(os.path.join(output_dir, "train.json"), "w"), indent=4)
    json.dump(val, open(os.path.join(output_dir, "val.json"), "w"), indent=4)
    json.dump(test, open(os.path.join(output_dir, "test.json"), "w"), indent=4)
    meta = {
        "language": language,
        "pdf_path": os.path.abspath(pdf_path) if pdf_path else None,
        "pdf_mtime": os.path.getmtime(pdf_path) if pdf_path and os.path.exists(pdf_path) else None,
        "num_samples": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
    }
    json.dump(meta, open(os.path.join(output_dir, "meta.json"), "w"), indent=4)

    print(f"[Dataset] Saved dataset to {output_dir}")


# ---------------------------
# High-level function
# ---------------------------
def build_dataset(
    pdf_path="data/raw_pdf/source_book.pdf",
    output_dir="data/processed",
    force_rebuild=False,
    language="es",
):
    if not force_rebuild:
        train_f = os.path.join(output_dir, "train.json")
        val_f = os.path.join(output_dir, "val.json")
        test_f = os.path.join(output_dir, "test.json")
        meta_f = os.path.join(output_dir, "meta.json")
        cached_ok = all(os.path.exists(f) for f in (train_f, val_f, test_f))
        if cached_ok and os.path.exists(meta_f):
            try:
                meta = json.load(open(meta_f))
                lang_ok = meta.get("language") == language
                pdf_ok = True
                if pdf_path and meta.get("pdf_path"):
                    try:
                        current_mtime = os.path.getmtime(pdf_path)
                        pdf_ok = abs(current_mtime - float(meta.get("pdf_mtime", current_mtime))) < 1e-3
                    except OSError:
                        pdf_ok = False
                if lang_ok and pdf_ok:
                    print(f"[Dataset] Found cached splits in {output_dir} with matching language/pdf, skipping rebuild.")
                    return json.load(open(train_f)), json.load(open(val_f)), json.load(open(test_f))
                else:
                    reason = "language mismatch" if not lang_ok else "PDF changed"
                    print(f"[Dataset] Cache invalidated ({reason}); rebuilding dataset.")
            except Exception:
                print("[Dataset] Cache metadata missing or corrupted; rebuilding dataset.")

    print(f"[Dataset] Using PDF: {pdf_path}")

    text = extract_pdf_text(pdf_path)

    extractive = build_extractive_qa(text, language=language)
    closed = build_closed_ended_qa(text, language=language)

    print("[Dataset] Total raw samples:", len(extractive) + len(closed))

    all_pairs = extractive + closed
    filtered = semantic_filtering(all_pairs)

    split_and_save(filtered, output_dir, language=language, pdf_path=pdf_path)

    return json.load(open(os.path.join(output_dir, "train.json"))), json.load(open(os.path.join(output_dir, "val.json"))), json.load(open(os.path.join(output_dir, "test.json")))

# ---------------------------
# Loader for evaluation
# ---------------------------
def load_validation_data(path="data/processed/val.json"):
    return json.load(open(path))


def load_train_data(path="data/processed/train.json"):
    return json.load(open(path))


def load_test_data(path="data/processed/test.json"):
    return json.load(open(path))
