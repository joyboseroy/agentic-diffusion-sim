# darshana-temporal-analysis

**Four computational studies of Indian philosophical traditions using the darshana-graph.**

This repository contains the code, data, and experiments supporting the paper:

> **Attribution Bias in Philosophical Knowledge Graphs: Corpus Frequency versus Temporal Sourcing**
> Joy Bose — arXiv:2026.XXXXX

The paper argues that corpus-frequency attribution systematically misattributes philosophical concepts to later, textually dominant schools, demonstrates this using the [darshana-graph](https://huggingface.co/datasets/joyboseroy/darshana-graph), and shows that temporal attribution enables a new class of cross-tradition structural analysis.

---

## Studies in this repository

### Study 1: Attribution Bias and Temporal Sourcing (`temporal/`)

Corpus-frequency attribution assigns each concept to the school that mentions it most in the available text. We show this conflates textual power, historical priority, and philosophical significance.

**Key findings:**
- Seven of the top 25 concepts by betweenness centrality predate their attributed school by 288 to 2,288 years
- Moksha, attributed to Advaita Vedanta, appears first in Jain sources over 1,200 years earlier
- The network at 300 BCE is 59% Vedic, 24% Jain, 18% Buddhist — not the Advaita-dominated picture corpus frequency produces
- Between 300 CE and 800 CE the network grows from 18 to 1,028 nodes with 97.4% carrying Advaita proxy dates — corpus composition artifact, not historical reality

```bash
cd temporal/
python3 temporal_analysis.py
python3 temporal_analysis.py --concept moksha
python3 temporal_analysis.py --era -300
python3 integrate_temporal_v2.py --era_betweenness
```

---

### Study 2: Structural Homology Across Traditions (`temporal/`)

Ego-network feature vectors applied to 48 temporally labelled concepts across eight traditions identify cross-tradition concept pairs with high structural similarity.

**Key findings:**
- 8/14 known scholarly correspondences recovered, including purusha-jiva (sim 0.990) and prakriti-maya (sim 0.972)
- Nibbana and samsara score 0.954 despite doctrinal opposition — both are the ultimate reference concept in their tradition's soteriology
- Cetana (Buddhist intention) and ajiva (Jain non-living matter) score 0.923 — absent from scholarly literature, identified computationally
- Moha (Buddhist delusion) and tapas (Jain austerity) score 0.959 — mirror-image soteriological positions

```bash
cd temporal/
pip install scikit-learn --break-system-packages
python3 structural_homology.py --top_n 50
```

---

### Study 3: Diffusion Simulation and Path-Finding (`notebooks/`)

How far does a new idea travel if seeded at a structural hub? Which historical figures were the real bridges between traditions?

**Key findings:**
- Centrality-based seeding reaches 62.6% adoption vs 58.4% for random after 25 steps
- Ramakrishna is the most central bridge figure: one step from nirvana, fana, and brahman simultaneously
- Shortest paths: sunyata → krsna consciousness (2 steps via maya), anatta → atman (2 steps via Ramana Maharshi)

```bash
pip install networkx matplotlib datasets --break-system-packages
python3 notebooks/darshana_diffusion_v2.py --compare --steps 25
python3 notebooks/darshana_transmission_viz.py --from anatta --to atman --animate
python3 notebooks/darshana_transmission_viz.py --show_figure Ramakrishna
```

---

### Study 4: Diachronic Sense Disambiguation (`diachronic/`)

Moves from a lexical graph (one node per word) toward a sense-disambiguated graph (one node per philosophical concept-sense). Addresses the core limitation identified by peer reviewers: earliest word attestation is not the same as earliest philosophical concept development. Maya in the Rigveda means magical power; maya as cosmic illusion enters Advaita via Gaudapada's adoption of Madhyamaka arguments (King 1995). The pipeline separates these senses computationally.

**Key findings:**
- 52 key concepts clustered into philosophical senses using TF-IDF/SVD + Groq LLM labelling
- Samsara: 4 senses distinguished (Mahayana cyclic suffering, Yogacara, early Vedic, general)
- Maya: 5 senses (Vedic magical power, Advaita cosmic illusion, Dvaita divine power, Mahayana, Yogacara)
- Avidya: 5 senses separating Buddhist (avijja) from Advaita usage
- Darshana-graph expanded from 28,322 to 45,155 edges by adding Mahayana Buddhist and Theravada sources with correct school labels
- Running betweenness centrality on the Buddhist-only graph (buddhist_edges.jsonl, 16,833 edges) produces a strikingly different top-25 from the original Vedanta-dominated graph. Pratityasamutpada ranks first (0.155), followed by prajna (0.141), sunyata (0.119), anatta (0.111), and nirvana (0.111). Brahman appears at rank 23 (0.013), present only as a concept Buddhist texts reference in critique. This contrast quantifies the corpus composition bias the paper documents.
- Jain-only betweenness centrality places jiva (0.286) at the network centre, followed by samyak darshana (0.135), ahimsa (0.130), and moksha (0.090) — the actual architecture of Jain soteriology. The three-way comparison of Hindu-only, Buddhist-only, and Jain-only betweenness demonstrates that the same graph methodology produces completely different philosophical maps depending on which tradition's texts supply the edges.

```bash
cd diachronic/

# Step 1: Extract occurrences with evidence quotes
python3 extract_occurrences_v2.py --input darshana_graph_v3.jsonl \
    --output occurrences_v3.json

# Step 2: Cluster into philosophical senses
python3 cluster_occurrences_v3.py --input occurrences_v3.json \
    --output occurrences_clustered_v3.json

# Step 3: Extract from new source texts (Mahayana PDFs)
python3 extract_texts.py \
    --dir /path/to/mahayana/pdfs \
    --school mahayana \
    --output mahayana_edges.jsonl

# Step 4: Extract from Theravada texts
python3 extract_texts.py \
    --dir /path/to/theravada/texts \
    --school theravada \
    --output pali_edges.jsonl

# Merge expanded graph
cat darshana_graph_original.jsonl mahayana_edges.jsonl pali_edges.jsonl \
    > darshana_graph_v3.jsonl

# Step 5: Extract from Jain canonical texts
python3 extract_jain.py \
    --dir /path/to/jain/pdfs \
    --output jain_edges.jsonl

# Build v4 with all traditions
cat darshana_graph_original.jsonl mahayana_edges.jsonl \
    pali_edges.jsonl jain_edges_clean.jsonl \
    > darshana_graph_v4.jsonl
```

---

## Repository structure

```
darshana-temporal-analysis/
├── temporal/                           # Studies 1 and 2
│   ├── temporal_source_layer.json      # 40 sources, ~120 concept-datings
│   ├── temporal_analysis.py            # Mismatch analysis and era snapshots
│   ├── integrate_temporal_v2.py        # Three-tier era betweenness analysis
│   ├── structural_homology.py          # Cross-tradition ego-network similarity
│   ├── homologues_v7.json              # Top-200 cross-tradition homologue pairs
│   ├── add_teacher_terms.py            # Adds Pali/Buddhist vocabulary to temporal layer
│   └── fix_temporal_layer.py           # Corrects maya, avidya, samsara attributions
├── diachronic/                         # Study 4
│   ├── extract_occurrences_v2.py       # Extract concept occurrences with evidence quotes
│   ├── cluster_occurrences_v3.py       # Cluster occurrences into philosophical senses
│   ├── extract_texts.py                # Extract edges from new source PDFs/texts
│   ├── fix_occurrences.py              # Merge Pali-Sanskrit equivalents
│   ├── extract_jain.py                 # Extract edges from Jain canonical PDFs
│   └── graph_loader.py                 # Load graph from local file or HuggingFace
├── notebooks/                          # Study 3
│   ├── darshana_diffusion_v2.py        # Darshana diffusion simulation
│   ├── darshana_path_viz.py            # Path finder on classical graph
│   ├── darshana_transmission_viz.py    # Two-layer path finder and animator
│   └── darshana_diffusion_standalone.py
├── agents/
│   └── belief_agent.py
├── network/
│   └── generator.py
├── simulation/
│   └── diffusion.py
├── visualization/
│   └── animate.py
├── transmission_layer.json             # 24 historical figures, 155 edges
├── run_sim.py
└── examples/
```

---

## Data sources

| Dataset | Location | Edges | Description |
|---|---|---|---|
| darshana-graph v1 | [HuggingFace](https://huggingface.co/datasets/joyboseroy/darshana-graph) | 28,322 | Original Vedantic commentary corpus |
| darshana-graph v2 | [HuggingFace](https://huggingface.co/datasets/joyboseroy/darshana-graph) | 44,390 | v1 + Mahayana Buddhist extraction |
| darshana-graph v3 | [HuggingFace](https://huggingface.co/datasets/joyboseroy/darshana-graph) | 45,155 | v1 + Mahayana + Theravada |
| mahayana_edges.jsonl | [HuggingFace](https://huggingface.co/datasets/joyboseroy/darshana-graph) | 16,068 | Mahayana extraction only (madhyamaka, yogacara, mahayana, chan) |
| pali_edges.jsonl | [HuggingFace](https://huggingface.co/datasets/joyboseroy/darshana-graph) | 765 | Theravada extraction only |
| occurrences_clustered_v3.json | [HuggingFace](https://huggingface.co/datasets/joyboseroy/darshana-graph) | — | 52 concepts sense-disambiguated |
| temporal_source_layer.json | `temporal/` | — | 40 scholarly sources, ~120 concept-datings |
| transmission_layer.json | repo root | — | 24 historical figures, 155 transmission edges |
| homologues_v7.json | `temporal/` | — | Top-200 cross-tradition structural homologue pairs |
| jain_edges.jsonl | [HuggingFace](https://huggingface.co/datasets/joyboseroy/darshana-graph) | 3,659 | Jain canonical extraction — Tattvarthasutra, Samayasara, Acaranga Sutra. School label: jainism. |
| non_vedic_edges.jsonl | [HuggingFace](https://huggingface.co/datasets/joyboseroy/darshana-graph) | 21,257 | Buddhist + Jain combined — for cross-tradition analysis without Vedantic corpus. |
| darshana-graph v4 | [HuggingFace](https://huggingface.co/datasets/joyboseroy/darshana-graph) | 48,814 | v1 + Mahayana + Theravada + Jain canonical. Most complete version. |

**Mahayana source texts extracted:** Heart Sutra (Hsuan Hua commentary), Mulamadhyamakakarika (Kalupahana), Lankavatara Sutra (Suzuki), Lotus Sutra, Vimalakirti Nirdesa Sutra, Bodhicaryavatara (Shantideva/Wallace), Platform Sutra, Treasury of Mahayana Sutras (Maharatnakuta), Surangama Sutra, Diamond Sutra commentary, Ksitigarbha Sutra, Three Pure Land Sutras.

**Theravada source texts extracted:** 15 text files covering Majjhima Nikaya and Samyutta Nikaya (Nidana-samyutta). Full Pali re-extraction with school labels is planned for v4.

**Jain source texts extracted:** Tattvarthasutra (Umasvami/Pujyapada, Vijay K. Jain translation), Samayasara (Kundakunda, Sacred Books of the Jainas), Acaranga Sutra (Mahendrakumar Muni translation), Agama Introduction Booklet.

---

## Related projects

| Project | Connection |
|---|---|
| [darshana-graph](https://huggingface.co/datasets/joyboseroy/darshana-graph) | Source dataset |
| [vada-simulator](https://github.com/joyboseroy/vada-simulator) | Multi-agent philosophical debate engine |
| [bengal-dharma-corpus](https://github.com/joyboseroy/bengal-dharma-corpus) | Computational study of Bengali religious lexicon |
| [digital-buddhism](https://github.com/joyboseroy/digital-buddhism) | Buddhist teacher network analysis on Reddit |
| [falkor-irac](https://github.com/joyboseroy/falkor-irac) | Graph-RAG legal reasoning |

---

## Citation

```bibtex
@article{bose2026attribution,
  author = {Bose, Joy},
  title = {Attribution Bias in Philosophical Knowledge Graphs:
           Corpus Frequency versus Temporal Sourcing},
  year = {2026},
  journal = {arXiv preprint},
  url = {https://arxiv.org/abs/2026.XXXXX}
}

@software{bose2026darshana,
  author = {Bose, Joy},
  title = {darshana-temporal-analysis: Computational Studies of
           Indian Philosophical Traditions},
  year = {2026},
  url = {https://github.com/joyboseroy/darshana-temporal-analysis}
}

@dataset{bose2025darshana,
  author = {Bose, Joy},
  title = {darshana-graph: A Knowledge Graph of Indian Philosophy},
  year = {2025},
  url = {https://huggingface.co/datasets/joyboseroy/darshana-graph}
}
```

---

## Medium article

[How Ideas Travel Across Religions: A Computational Journey Through 2,000 Years of Indian Philosophy](https://joyboseroy.medium.com/how-ideas-travel-across-religions-a-computational-journey-through-2000-years-of-indian-philosophy-ce91b089e888)
