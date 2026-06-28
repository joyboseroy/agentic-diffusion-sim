"""
extract_texts.py
================
Generic extraction script for philosophical texts.
Handles PDF and TXT files with explicit school label.

Run:
    # Theravada Pali texts
    python3 extract_texts.py \
        --dir /mnt/c/Users/ebosjoy/Downloads/theravada \
        --school theravada \
        --output pali_edges.jsonl

    # Or single file
    python3 extract_texts.py \
        --file thera1.txt \
        --school theravada \
        --output pali_edges.jsonl
"""

import json, os, argparse, time, re
from pathlib import Path

GROQ_MODEL = "llama-3.3-70b-versatile"

EXTRACTION_PROMPT = """You are extracting philosophical concept relationships from a {school} Buddhist text.

Text passage:
{passage}

Extract up to 8 philosophical concept pairs. For each:
- concept_a: first concept (use standard Pali/Sanskrit term)
- concept_b: second concept
- relation: one of [IS_IDENTICAL_TO, IS_DISTINCT_FROM, IS_CAUSE_OF, LEADS_TO,
  PRESUPPOSES, SUBLATES, OBSTRUCTS, IS_MANIFESTATION_OF, IS_QUALIFIED_ASPECT_OF,
  CONTRADICTS_IN_SCHOOL]
- evidence_quote: exact phrase showing this relationship (max 100 chars)
- confidence: high/medium/low

For Theravada texts focus on: dukkha, anicca, anatta, nibbana, tanha, vedana, 
cetana, sila, samadhi, prajna/panna, nirodha, magga, kamma, dhamma, citta, 
mano, phassa, upadana, bhava, jati, dependent origination links, 
four noble truths, eightfold path, five aggregates/skandhas, six sense bases.

For Mahayana texts focus on: sunyata, prajna, bodhicitta, karuna, upaya,
tathagatagarbha, tathata, dharmadhatu, alayavijnana, paramita, bodhisattva,
two truths, svabhava, pratityasamutpada, buddha-nature.

Return ONLY a JSON array. If no philosophical pairs found, return [].
Example:
[
  {{"concept_a": "tanha", "concept_b": "dukkha",
    "relation": "IS_CAUSE_OF",
    "evidence_quote": "craving is the origin of suffering",
    "confidence": "high"}}
]"""

def chunk_text(text, chunk_size=600, overlap=80):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

def read_file(path):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            return "\n".join(
                p.extract_text() or "" for p in reader.pages
            )
        except Exception as e:
            print(f"  PDF error: {e}")
            return ""
    else:
        try:
            return Path(path).read_text(errors="replace")
        except Exception as e:
            print(f"  Text error: {e}")
            return ""

def extract_from_chunk(chunk, school, groq_client):
    prompt = EXTRACTION_PROMPT.format(
        school=school, passage=chunk[:2000])
    try:
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800, temperature=0.1)
        text = r.choices[0].message.content.strip()
        text = re.sub(r"```json|```", "", text).strip()
        edges = json.loads(text)
        return edges if isinstance(edges, list) else []
    except json.JSONDecodeError:
        return []
    except Exception as e:
        print(f"    Groq error: {e}")
        time.sleep(2)
        return []

def process_file(path, school, groq_client, limit=None, skip_pages=0):
    filename = Path(path).stem
    print(f"\n  [{filename}] school={school}")

    text = read_file(path)
    if not text.strip():
        print(f"  Empty — skipping")
        return []

    # Skip front matter (e.g. intro pages converted to words)
    if skip_pages > 0:
        # Approximate: skip first skip_pages*300 words
        words = text.split()
        text = " ".join(words[skip_pages*300:])

    chunks = chunk_text(text)
    print(f"  Chunks: {len(chunks)}")
    if limit:
        chunks = chunks[:limit]

    all_edges = []
    for i, chunk in enumerate(chunks):
        edges = extract_from_chunk(chunk, school, groq_client)
        for e in edges:
            e["school"]  = school
            e["source"]  = filename
            e["chunk_id"] = i
            e["concept_a"] = str(e.get("concept_a","")).strip().lower()
            e["concept_b"] = str(e.get("concept_b","")).strip().lower()
        edges = [e for e in edges
                 if e["concept_a"] and e["concept_b"]
                 and e["concept_a"] != e["concept_b"]
                 and len(e["concept_a"]) > 2
                 and len(e["concept_b"]) > 2]
        all_edges.extend(edges)
        if (i+1) % 10 == 0:
            print(f"  Chunk {i+1}/{len(chunks)} — edges: {len(all_edges)}")
        time.sleep(0.3)

    print(f"  Done: {len(all_edges)} edges from {filename}")
    return all_edges

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",    default=None)
    parser.add_argument("--file",   default=None)
    parser.add_argument("--school", required=True,
        help="School label: theravada / madhyamaka / mahayana / chan / yogacara")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit",  type=int, default=None,
        help="Max chunks per file (for testing)")
    parser.add_argument("--skip_pages", type=int, default=0,
        help="Approximate pages of front matter to skip")
    args = parser.parse_args()

    key = os.environ.get("GROQ_API_KEY","")
    if not key:
        print("ERROR: set GROQ_API_KEY"); return
    from groq import Groq
    client = Groq(api_key=key)
    print(f"Groq ready | school={args.school}")

    # Collect files
    if args.file:
        files = [args.file]
    elif args.dir:
        d = Path(args.dir)
        files = sorted(list(d.glob("*.txt")) +
                       list(d.glob("*.pdf")) +
                       list(d.glob("*.html")))
    else:
        print("ERROR: specify --file or --dir"); return

    print(f"Files: {len(files)}")

    # Load existing output to allow resuming
    existing = []
    existing_sources = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    existing.append(e)
                    existing_sources.add(e.get("source",""))
        print(f"Resuming: {len(existing)} edges already in {args.output}")

    all_edges = list(existing)
    with open(args.output, "a") as out_f:
        for path in files:
            source = Path(path).stem
            if source in existing_sources:
                print(f"  SKIP {source} (already done)")
                continue
            edges = process_file(
                path, args.school, client,
                limit=args.limit,
                skip_pages=args.skip_pages)
            for e in edges:
                out_f.write(json.dumps(e) + "\n")
            out_f.flush()
            all_edges.extend(edges)

    # Summary
    from collections import Counter
    print(f"\n{'='*55}")
    print(f"Total edges: {len(all_edges)}")
    concept_counts = Counter()
    for e in all_edges:
        concept_counts[e["concept_a"]] += 1
        concept_counts[e["concept_b"]] += 1
    print(f"\nTop 25 concepts:")
    for c, n in concept_counts.most_common(25):
        print(f"  {c:<35}: {n:>5}")

if __name__ == "__main__":
    main()
