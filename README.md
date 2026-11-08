# Supervised Constrastive Vision–Language Alignment and Retrieval Model with Evolutionary Semantic Language Expansion (CLIP-Style CIFAR-100)

Built a lightweight image–text alignment and retrieval model for CIFAR-100 by training Skip-gram word embeddings from a co-occurrence network built on Visual Genome to semantically capture visually-grounded object scene descriptions.
Demonstrated stable cross-domain embedding transfer learning by expanding the text embedding space using (1+λ) evolutionary algorithm, coherently integrating CIFAR-100 class labels for alignment with image embeddings while preserving the learned semantic relationships - without retraining the entire latent space.
Used a pretrained MobileNetV3 as the vision backbone and aligned image features to the EA-enriched Skip-gram text space using CLIP-style supervised multimodal contrastive learning with InfoNCE loss, achieving ~91% Recall@10 in image-to-text retrieval and improved semantic coherence and generalization across class clusters.

The system achieves **~91% Recall@10** on image→text retrieval and demonstrates strong class-wise semantic alignment, stable neighbourhood structures, and meaningful OOD retrieval behaviour.

---

## Table of Contents

- [Overview](#-project-overview)
- [Architecture](#-system-architecture)
- [Training Pipeline](#-training-pipeline)
- [Evaluation & Results](#-evaluation--results)
- [Future Work](#-future-work)

---

## Project Overview

This project explores **vision–language alignment without using large pretrained text encoders**.

Instead:

1. A **Skip-Gram model** is trained on a Visual Genome co-occurrence network  
2. CIFAR-100 class labels are inserted into the embedding space using an  
   **Evolutionary Anchor-based semantic insertion method (Weighted-Anchor (1+lambda) GA Insertion)**
3. A frozen CNN image encoder (MobileNetV3) is trained with a  
   **projection head + symmetric InfoNCE loss**  
   to align images with semantically coherent word embeddings

The result is a **structured multimodal embedding space** that is:

- visually grounded  
- semantically interpretable  
- robust across classes and unseen examples

   Nuance: The Skip-Gram model learns semantic relationships from Visual Genome co-occurrence data, and an evolutionary anchor-insertion process integrates CIFAR-100 class labels into this space. The projection head is then trained to align CNN image embeddings to these semantically-enriched word representations, enabling the model to map images into a concept space shaped by how real-world scenes are described.

---

## System Architecture

### Text Embedding Backbone — Skip-Gram over Visual Genome

Built a semantic text backbone by training Skip-Gram word embeddings on a word co-occurrence graph constructed from Visual Genome region descriptions — 5.4M short sentences that people wrote about regions of real images (e.g. *"a white car parked on the street"*).

- Constructed a co-occurrence graph from region descriptions: each word is a node, and two words are linked when they appear **directly next to each other** in a description (455 words, 48,764 weighted links).
- Learned 64-dimensional embeddings using a custom Skip-Gram model with negative sampling.
- Words that are **described in similar ways** end up with similar vectors (e.g. cat–dog 0.92, car–truck 0.90, table–chair 0.86 cosine similarity).
- This gives an embedding space built only from descriptions of images, rather than a generic language-model vocabulary trained on books or web text.

#### What the graph represents

The graph represents **how scenes are described in language**: which words people use together when describing what they see.

For example, *"a white car parked on the street"* adds the links `white–car`, `car–parked`, `parked–on` and `the–street`. Summed over all 5.4M descriptions, each word's links and their counts form a profile of how that word is used to describe visual scenes.

Skip-Gram turns these profiles into vectors, so words with **similar profiles** become close in the embedding space, even if they rarely appear side by side. `cat` and `dog`, for instance, are both described as "sitting", "laying" and "brown", so their vectors are close (0.92 cosine similarity). This is **distributional semantics** ("words used in similar ways have similar meanings") learned from visual scene descriptions.

This suits the retrieval task: an image is matched to the class label whose vector is closest, so semantically similar labels sit near each other and near-misses stay meaningful.

**Why a graph instead of training Skip-Gram on the raw sentences:** the graph adds up neighbour counts across all 5.4M descriptions into one weighted network. That makes each word's neighbourhood explicit and easy to inspect (for example with the Neo4j queries in [graph_store.py](graph_store.py)), and lets Skip-Gram weight frequent pairs more heavily.

---

### 2) Evolutionary Weighted-Anchor Insertion — Cross-Domain Embedding Expansion

Expanded the VG embedding space to integrate CIFAR-100 class labels using a (1+λ) evolutionary anchor-insertion strategy instead of retraining from scratch. CIFAR-100 labels were inserted into the embedding space using an **evolutionary search with weighted-anchors**, instead of random or mean-based placement.

- New class vectors were inserted by evolving around selected semantic anchor words
- Fitness preserved semantic local neighbourhood structure (embedding space geometry)
- Avoided embedding drift and catastrophic disruption
- Enabled stable cross-domain transfer learning

This allowed CIFAR-100 labels to be added while:

✔ preserving existing semantics
✔ maintaining neighbourhood coherence
✔ avoiding costly full-model retraining

EA optimisation preserved:

- local semantic neighbourhoods  
- cosine-similarity structure  
- class-cluster continuity

Examples:

- reptiles form a stable cluster (lizard, crocodile, snake)
- vehicle classes align together
- trees + plants form meaningful hierarchies

This produces a **523-word enriched semantic vocabulary space**.

---

### 3) Vision Backbone — MobileNetV3 + InfoNCE Projection Head

- Backbone: **MobileNetV3-Small (frozen)**
- Output features: 576-dim
- MobileNetV3 extracts visual features
- Trainable projection head maps features -> text embedding space. Projection head learns alignment transformation.
- Used pretrained MobileNetV3 as a frozen visual encoder and added a trainable projection head to map image features into the Skip-Gram text embedding space.
- Embeddings are normalized for cross-modal similarity learning

This produced a lightweight and efficient VLM-style architecture.

Training objective:

✔ symmetric **InfoNCE contrastive loss**  
✔ aligns image & text embeddings  
✔ pushes apart negative pairs

This encourages:

- cross-modal consistency
- interpretable clusters
- robust image→text retrieval

---

### 4) Multimodal Alignment — CLIP-Style Contrastive Learning

- Aligned visual and text embeddings using supervised multimodal contrastive learning with InfoNCE loss.
- Images mapped to text-embedding space
- Matching pairs pulled together
- Non-matching pairs pushed apart
- Training objective encourages semantic alignment

This enabled both:

✔ zero-shot image-to-text retrieval
✔ text-guided image reasoning

## Training Pipeline (End-to-End)
Visual Genome captions → build co-occurrence graph → Train Skip-Gram word embeddings → Evolutionary Anchor Insertion for CIFAR-100 labels → Freeze text embedding space (semantic prior) → Train projection head using contrastive InfoNCE → Evaluate retrieval, alignment & OOD behaviour

The projection head learns to **adapt image features into a fixed, semantically structured space**, rather than learning text + image jointly — reducing overfitting risk and improving interpretability.

---

## Neo4j Graph Storage

The co-occurrence graph can be stored in Neo4j ([graph_store.py](graph_store.py)) instead of being rebuilt from the Visual Genome text on every run. Neo4j keeps the permanent copy; training still runs on an in-memory NetworkX copy loaded from it, so the trained checkpoints stay valid and nothing needs retraining.

```bash
pip install -r requirements.txt
export NEO4J_PASSWORD=<choose one, 8+ chars>
docker compose up -d          # Neo4j Browser at http://localhost:7474
python graph_store.py         # unit tests (Neo4j tests run when NEO4J_PASSWORD is set)
```

In the notebook, Stage 1 uses Neo4j when `NEO4J_PASSWORD` is set: it builds and saves the graph the first time, then loads it from Neo4j. Without `NEO4J_PASSWORD` it builds the graph in memory exactly as before. Stage 2 checks that the stored vocabulary is in the same order as the Skip-gram checkpoint's embedding rows, and Stage 2b lists hub words and how much of the vocabulary each word's Skip-gram context covers.

---

## Evaluation & Results

Evaluation is retrieval-based, not classification-based.

Skip-Gram Evaluation:
- Pairwise Cosine similarity 
- Nearest Neighbour Analysis
- Word Analogies

Projection Head Evaluation:
- Image->Text Retieval Recall@K
- Text->Image Recall@K
- Confusion Matrix (to evaluate k-NN (zero shot) classification on CIFAR-100 images)
- Per-class similarity distribution
- OOD retrieval behaviour

### Achieved Performance

- **~91% Recall@10 (image->text retrieval)**
- **~100% Recall@10 (text->image retrieval)**
- Strong alignment in vehicles, plants, structures, tools
- Meaningful class clusters & similarity hierarchies
<img width="818" height="802" alt="image" src="https://github.com/user-attachments/assets/040ca022-d37c-476a-9a9b-9b188be4d87d" />
<img width="1122" height="992" alt="image" src="https://github.com/user-attachments/assets/216b7749-b4fa-4593-b711-db8d26f8922f" />
<img width="1558" height="1204" alt="image" src="https://github.com/user-attachments/assets/fbb432bb-16fb-4c88-828d-9a607bda82d5" />

This indicates:

- stable semantic geometry
- visually grounded embedding relationships
- cross-modal correspondence

## Future Work

Planned extensions:

- Compare model performance against standard cross-entropy classification version of this use-case
- Zero-shot learning with relational prompts (vanilla CLIP-style)
  - Implement Inductive Zero-Shot Learning (IZSL) - Train on seen classes, test on completely unseen classes at inference (no access to unseen class data during training)
  - Implement Transductive Zero-Shot Learning (TZSL) - Train on seen classes, but have access to unlabeled images from unseen classes during training
- Extend the graph with scene-level structure in Neo4j: link objects that appear in the same image (Visual Genome `objects.json`) and add typed relations such as `car —on→ road` (`relationships.json`) alongside the existing word links. These can be stored and queried without retraining, then used to retrain the embeddings on the combined graph (Skip-Gram) or to learn the typed relations with knowledge-graph embeddings (e.g. TransE)
- Tune the Skip-Gram context size and filter high-frequency function words from the graph
- Joint training of both modalities
- Few-shot novel class embedding insertion
- Graph neural networks over embedding neighbourhoods
- Compare against CLIP text encoder representations
- Instead of manually inserting the CIFAR-100 label embeddings in the text (Skip-Gram) embedding space using EA, try to achieve the system to predict new (CIFAR-100 labels) embeddings and position them semantically in the text embedding space (refer to VL-JEPA - it semantically predicts new embeddings in the embedding space level). Reference: [https://doi.org/10.48550/arXiv.2512.10942]

---
