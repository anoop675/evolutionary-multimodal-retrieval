"""
Neo4j storage for the Visual Genome word co-occurrence graph.

Neo4j holds the permanent copy of the graph. Training still runs on an
in-memory NetworkX copy: load_graph() rebuilds exactly the dict that
process_text_network() + filter_punctuation_from_network() produce
('graph', 'nodes', 'distance_matrix', ...), so train_embeddings() needs no changes.

Graph model:
    (:Graph {graph_id, num_nodes, num_edges, no_edge_distance, ...})   one per stored graph
    (:Word  {graph_id, name, index, freq})                             one per vocabulary word
    (:Word)-[:CO_OCCURS {count, distance}]->(:Word)                    stored once per pair, query undirected

Connection settings come from environment variables:
    NEO4J_URI (default bolt://localhost:7687), NEO4J_USER (default neo4j), NEO4J_PASSWORD (required)
"""

import os
import unittest
from collections import Counter
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import torch


# ============================================================================
# Connection
# ============================================================================

def get_driver(uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
    """Open a Neo4j driver, reading credentials from environment variables by default."""
    from neo4j import GraphDatabase  # imported here so the offline helpers work without the neo4j package
    uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.environ.get("NEO4J_USER", "neo4j")
    password = password or os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise ValueError("Set the NEO4J_PASSWORD environment variable (never hard-code it).")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()  # fail fast if the server is down or credentials are wrong
    return driver


def ensure_schema(driver):
    """Create uniqueness constraints (also act as indexes for fast lookups)."""
    with driver.session() as s:
        s.run("CREATE CONSTRAINT graph_id_unique IF NOT EXISTS "
              "FOR (g:Graph) REQUIRE g.graph_id IS UNIQUE")
        s.run("CREATE CONSTRAINT word_unique IF NOT EXISTS "
              "FOR (w:Word) REQUIRE (w.graph_id, w.name) IS UNIQUE")  # same word may exist in several stored graphs
        s.run("CREATE INDEX word_index IF NOT EXISTS FOR (w:Word) ON (w.graph_id, w.index)")


# ============================================================================
# Save / load
# ============================================================================

def graph_exists(driver, graph_id: str) -> bool:
    """Return True if a graph with this id is already stored."""
    with driver.session() as s:
        rec = s.run("MATCH (g:Graph {graph_id: $g}) RETURN count(g) AS n", g=graph_id).single()
    return rec["n"] > 0


def delete_graph(driver, graph_id: str, batch_size: int = 10000):
    """Remove a stored graph (its Graph node, Word nodes and edges)."""
    with driver.session() as s:
        s.run("MATCH (g:Graph {graph_id: $g}) DELETE g", g=graph_id)
        while True:  # delete in batches so big graphs don't blow the transaction memory
            rec = s.run("MATCH (w:Word {graph_id: $g}) WITH w LIMIT $b DETACH DELETE w RETURN count(*) AS n",
                        g=graph_id, b=batch_size).single()
            if rec["n"] == 0:
                break


def _no_edge_distance(graph: nx.Graph, nodes: List[str], distance_matrix: np.ndarray) -> Optional[float]:
    """
    Distance value shared by every pair of words that never co-occurred.

    compute_distance_matrix() gives non-adjacent pairs distance (max_count + 1), so one number
    is enough to rebuild the full dense matrix from the edges. Raises if that assumption breaks.
    """
    adj = nx.to_numpy_array(graph, nodelist=nodes, weight=None) != 0  # True where an edge exists
    non_edge = ~adj
    np.fill_diagonal(non_edge, False)  # the diagonal is always 0, not a "non-edge"
    if not non_edge.any():
        return None  # complete graph: every pair is an edge
    values = distance_matrix[non_edge]
    if not np.allclose(values, values[0]):
        raise ValueError("Non-adjacent pairs have different distances; the matrix cannot be rebuilt from edges.")
    return float(values[0])


def save_graph(driver, network_data: Dict, graph_id: str, metadata: Optional[Dict] = None,
               overwrite: bool = False, batch_size: int = 5000) -> Dict:
    """
    Store a co-occurrence graph in Neo4j.

    Args:
        network_data: dict from process_text_network() (+ filter_punctuation_from_network()).
                      Uses 'graph', 'nodes', 'distance_matrix' and optionally 'token_counts'.
        graph_id: name for this graph, e.g. "vg_rare0.00025". Lets several versions coexist.
        metadata: extra settings to record on the Graph node (rare_threshold, source, ...).
        overwrite: replace an existing graph with the same id.
    Returns:
        summary dict with node/edge counts.
    """
    graph, nodes, dm = network_data["graph"], network_data["nodes"], network_data["distance_matrix"]
    token_counts = network_data.get("token_counts") or {}

    if set(graph.nodes()) != set(nodes):
        raise ValueError("graph nodes and 'nodes' list differ; filter both together before saving.")
    if dm.shape != (len(nodes), len(nodes)):
        raise ValueError(f"distance_matrix shape {dm.shape} does not match {len(nodes)} nodes.")

    ensure_schema(driver)
    if graph_exists(driver, graph_id):
        if not overwrite:
            raise ValueError(f"Graph '{graph_id}' already exists; pass overwrite=True to replace it.")
        delete_graph(driver, graph_id)

    no_edge = _no_edge_distance(graph, nodes, dm)
    idx = {w: i for i, w in enumerate(nodes)}
    max_dist = no_edge if no_edge is not None else float(dm.max()) + 1.0  # reference for turning distance back into a count

    node_rows = [{"name": w, "index": i, "freq": int(token_counts.get(w, 0))} for i, w in enumerate(nodes)]  # index = embedding row
    edge_rows = []
    for u, v in graph.edges():
        a, b = sorted((idx[u], idx[v]))  # store each undirected pair once, low index -> high index
        d = float(dm[a, b])
        edge_rows.append({"a": a, "b": b, "distance": d, "count": max_dist - d})  # count = symmetrised co-occurrence count

    with driver.session() as s:
        s.run("CREATE (g:Graph {graph_id: $g}) SET g += $props, g.created_at = datetime()",
              g=graph_id, props={**(metadata or {}),
                                 "num_nodes": len(nodes),
                                 "num_edges": len(edge_rows),
                                 "no_edge_distance": no_edge,
                                 "distance_mode": network_data.get("distance_mode", "inverted")})
        for i in range(0, len(node_rows), batch_size):
            s.run("UNWIND $rows AS r CREATE (:Word {graph_id: $g, name: r.name, index: r.index, freq: r.freq})",
                  g=graph_id, rows=node_rows[i:i + batch_size])
        for i in range(0, len(edge_rows), batch_size):
            s.run("UNWIND $rows AS r "
                  "MATCH (a:Word {graph_id: $g, index: r.a}), (b:Word {graph_id: $g, index: r.b}) "  # uses the (graph_id, index) index
                  "CREATE (a)-[:CO_OCCURS {count: r.count, distance: r.distance}]->(b)",
                  g=graph_id, rows=edge_rows[i:i + batch_size])

    print(f"Saved graph '{graph_id}' to Neo4j: {len(nodes):,} words, {len(edge_rows):,} edges")
    return {"graph_id": graph_id, "num_nodes": len(nodes), "num_edges": len(edge_rows)}


def load_graph(driver, graph_id: str) -> Dict:
    """
    Load a stored graph back into memory.

    Returns the same keys train_embeddings() reads from process_text_network() output:
    'graph' (nx.Graph), 'nodes' (list, in embedding-row order), 'distance_matrix' (n x n),
    plus 'token_counts' (Counter) and 'metadata' (dict).
    """
    with driver.session() as s:
        meta = s.run("MATCH (g:Graph {graph_id: $g}) RETURN properties(g) AS p", g=graph_id).single()
        if meta is None:
            raise KeyError(f"No graph '{graph_id}' in Neo4j.")
        meta = dict(meta["p"])
        words = s.run("MATCH (w:Word {graph_id: $g}) RETURN w.name AS name, w.index AS index, w.freq AS freq "
                      "ORDER BY w.index", g=graph_id).data()  # ORDER BY index: Neo4j does not keep insertion order
        edges = s.run("MATCH (a:Word {graph_id: $g})-[r:CO_OCCURS]->(b:Word) "
                      "RETURN a.index AS a, b.index AS b, r.distance AS d", g=graph_id).data()

    nodes = [w["name"] for w in words]
    if [w["index"] for w in words] != list(range(len(nodes))):
        raise ValueError("Stored word indices are not 0..n-1; the vocabulary order is broken.")

    n = len(nodes)
    fill = meta.get("no_edge_distance")
    dm = np.full((n, n), fill if fill is not None else 0.0, dtype=np.float64)  # every pair starts as "never co-occurred"
    np.fill_diagonal(dm, 0.0)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for e in edges:
        dm[e["a"], e["b"]] = dm[e["b"], e["a"]] = e["d"]  # keep the matrix symmetric
        G.add_edge(nodes[e["a"]], nodes[e["b"]])

    print(f"Loaded graph '{graph_id}' from Neo4j: {n:,} words, {G.number_of_edges():,} edges")
    return {
        "graph": G,
        "nodes": nodes,
        "distance_matrix": dm,
        "token_counts": Counter({w["name"]: w["freq"] for w in words}),
        "metadata": meta,
    }


# ============================================================================
# Checks
# ============================================================================

def verify_roundtrip(original: Dict, loaded: Dict) -> bool:
    """Check that a loaded graph is identical to the one that was saved."""
    same_nodes = list(original["nodes"]) == list(loaded["nodes"])  # order matters: it is the embedding row order
    same_edges = ({frozenset(e) for e in original["graph"].edges()} ==
                  {frozenset(e) for e in loaded["graph"].edges()})
    same_dist = same_nodes and np.allclose(original["distance_matrix"], loaded["distance_matrix"])
    print(f"Round-trip check: nodes={same_nodes}, edges={same_edges}, distance_matrix={same_dist}")
    return same_nodes and same_edges and same_dist


def check_against_checkpoint(nodes: List[str], checkpoint_path: str) -> Dict:
    """
    Compare a graph's vocabulary with the vocabulary a trained Skip-gram checkpoint was built on.

    The checkpoint's embedding row i belongs to checkpoint['vocab'][i], so the stored graph only
    matches the trained model if its words appear in the same order.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocab = list(ckpt["vocab"])
    inserted = set(ckpt.get("inserted_words", []))

    prefix = 0
    for a, b in zip(nodes, vocab):  # how many leading words line up exactly
        if a != b:
            break
        prefix += 1

    graph_set, vocab_set = set(nodes), set(vocab)
    result = {
        "graph_words": len(nodes),
        "checkpoint_words": len(vocab),
        "matching_prefix": prefix,
        "order_matches": vocab[:len(nodes)] == list(nodes),
        "only_in_checkpoint": sorted(vocab_set - graph_set),
        "only_in_graph": sorted(graph_set - vocab_set),
        "inserted_words_in_graph": sorted(inserted & graph_set),  # words the checkpoint says were added by the EA
    }
    print(f"Checkpoint check: graph={len(nodes)} words, checkpoint={len(vocab)} words, "
          f"first {prefix} in the same order, order_matches={result['order_matches']}")
    print(f"  only in checkpoint ({len(result['only_in_checkpoint'])}): {result['only_in_checkpoint'][:15]}")
    print(f"  only in graph ({len(result['only_in_graph'])}): {result['only_in_graph'][:15]}")
    return result


# ============================================================================
# Analysis
# ============================================================================

def top_hubs(driver, graph_id: str, top_k: int = 20) -> List[Dict]:
    """Words with the most distinct neighbours (degree) and total co-occurrence count (strength)."""
    with driver.session() as s:
        return s.run(
            "MATCH (w:Word {graph_id: $g})-[r:CO_OCCURS]-() "  # no arrow: count edges in both directions
            "WITH w, count(r) AS degree, sum(r.count) AS strength "
            "MATCH (meta:Graph {graph_id: $g}) WITH w, degree, strength, meta.num_nodes AS n "  # vocab size stored at save time
            "RETURN w.name AS word, w.freq AS freq, degree, round(1.0 * degree / (n - 1), 3) AS degree_share, strength "
            "ORDER BY degree DESC LIMIT $k", g=graph_id, k=top_k).data()


def neighbours(driver, graph_id: str, word: str, top_k: int = 15) -> List[Dict]:
    """A word's direct neighbours, strongest co-occurrence first."""
    with driver.session() as s:
        return s.run(
            "MATCH (w:Word {graph_id: $g, name: $word})-[r:CO_OCCURS]-(c:Word) "
            "RETURN c.name AS word, r.count AS count ORDER BY r.count DESC LIMIT $k",
            g=graph_id, word=word, k=top_k).data()


def hop_reach(graph: nx.Graph, nodes: List[str], max_hops: int = 4) -> Dict[int, Dict[str, float]]:
    """
    For each hop limit k, how much of the vocabulary falls inside a word's Skip-gram context.

    Uses nx.single_source_shortest_path_length, the same call SkipGramDataset._build_contexts
    makes, so the numbers are exactly the contexts training sees. 'negative_pool' is how many
    words remain to sample negatives from (vocab - self - context).
    """
    n = len(nodes)
    reach = {k: [] for k in range(1, max_hops + 1)}
    for node in nodes:
        lengths = nx.single_source_shortest_path_length(graph, node, cutoff=max_hops)  # one BFS gives every k
        dists = np.array([d for d in lengths.values() if d > 0])
        for k in reach:
            reach[k].append(int((dists <= k).sum()))
    out = {}
    for k, vals in reach.items():
        vals = np.array(vals)
        out[k] = {
            "mean_context": float(vals.mean()),
            "median_share": float(np.median(vals) / (n - 1)),  # share of the other words inside the context
            "min_negative_pool": int((n - 1) - vals.max()),
            "words_with_empty_pool": int(((n - 1) - vals == 0).sum()),  # these hit the "any word" fallback in __getitem__
        }
    return out


def print_hop_reach(stats: Dict[int, Dict[str, float]], n_words: int):
    """Pretty-print hop_reach() output."""
    print(f"\nContext size vs vocabulary ({n_words} words)")
    print(f"  {'hops':>4}  {'mean context':>12}  {'median share':>12}  {'min neg pool':>12}  {'empty pools':>11}")
    for k, st in stats.items():
        print(f"  {k:>4}  {st['mean_context']:>12.1f}  {st['median_share']:>11.1%}  "
              f"{st['min_negative_pool']:>12}  {st['words_with_empty_pool']:>11}")


# ============================================================================
# Tests (Neo4j tests are skipped unless NEO4J_PASSWORD is set)
# ============================================================================

def _tiny_network():
    """Small network built with the real pipeline functions."""
    from collections import Counter as C
    from network_pipeline import build_unweighted_graph, compute_distance_matrix
    adj = C({("red", "car"): 3, ("car", "road"): 1, ("road", "car"): 1, ("tree", "forest"): 2})
    nodes = ["car", "red", "road", "tree", "forest", "lonely"]
    G = build_unweighted_graph(nodes, adj)
    dm, _ = compute_distance_matrix(nodes, adj, "inverted")
    return {"graph": G, "nodes": nodes, "distance_matrix": dm,
            "token_counts": C({"car": 5, "red": 3, "road": 2, "tree": 2, "forest": 2, "lonely": 1})}


class TestOffline(unittest.TestCase):
    def test_no_edge_distance(self):
        net = _tiny_network()
        self.assertEqual(_no_edge_distance(net["graph"], net["nodes"], net["distance_matrix"]), 2.5)  # symmetrised max count (3+0)/2 = 1.5, plus 1

    def test_hop_reach(self):
        net = _tiny_network()
        stats = hop_reach(net["graph"], net["nodes"], max_hops=2)
        self.assertEqual(stats[1]["min_negative_pool"], 5 - 2)  # 'car' touches red and road
        self.assertEqual(stats[2]["words_with_empty_pool"], 0)


@unittest.skipUnless(os.environ.get("NEO4J_PASSWORD"), "NEO4J_PASSWORD not set")
class TestNeo4jRoundTrip(unittest.TestCase):
    GID = "__unittest_tiny__"

    def setUp(self):
        self.driver = get_driver()

    def tearDown(self):
        delete_graph(self.driver, self.GID)
        self.driver.close()

    def test_roundtrip(self):
        net = _tiny_network()
        save_graph(self.driver, net, self.GID, overwrite=True)
        loaded = load_graph(self.driver, self.GID)
        self.assertTrue(verify_roundtrip(net, loaded))
        self.assertEqual(loaded["token_counts"]["car"], 5)

    def test_refuses_overwrite(self):
        net = _tiny_network()
        save_graph(self.driver, net, self.GID, overwrite=True)
        with self.assertRaises(ValueError):
            save_graph(self.driver, net, self.GID)

    def test_hubs_and_neighbours(self):
        save_graph(self.driver, _tiny_network(), self.GID, overwrite=True)
        self.assertEqual(top_hubs(self.driver, self.GID, 1)[0]["word"], "car")
        self.assertEqual(neighbours(self.driver, self.GID, "car")[0]["word"], "red")  # count 1.5 beats road's 1.0


def run_tests():
    print("=" * 70)
    print("RUNNING GRAPH STORE UNIT TESTS")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromModule(__import__("graph_store"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    exit(0 if run_tests() else 1)
