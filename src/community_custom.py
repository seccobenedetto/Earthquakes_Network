"""
Community detection suite for the Abe-Suzuki earthquake network.

Five methods, all returning the same {node: community_id} dict so they can be
passed to any downstream function interchangeably:

  Louvain              – modularity optimisation via leidenalg/igraph (Leiden
                         algorithm); strictly better than the NetworkX implementation
  Consensus Louvain    – 100-run co-occurrence → consensus matrix → Louvain;
                         removes partition instability inherent to single-run Louvain
  Spectral             – k-way spectral clustering on the normalised Laplacian
                         (Jordan-Weiss); k taken from Louvain community count
  InfoMap              – flow-based compression (directed, weighted); identifies
                         communities as regions where random walkers stay trapped
  HDBSCAN-Geographic   – density-based clustering on projected (x, y) node
                         coordinates; communities = spatial density concentrations
                         independent of network topology

NMI utilities compare any pair of partitions; the heatmap shows pairwise
agreement across all methods.

Partition scoring: ``score_partition`` computes nine quality metrics for a
single partition (modularity Q, conductance, coverage, Ncut, map equation,
DC-SBM log-likelihood, Surprise, geographic compactness, depth coherence).
``compare_partitions`` scores all methods and returns a tidy DataFrame;
``plot_partition_scores`` renders a z-score-normalised heatmap ranked by Q.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize

from src.plotutils import savefig, save_plotly, _slug, pres_title

from src.network import discretize_space_3d
from src.network_custom import build_abe_suzuki_network_custom_hybrid

log = logging.getLogger(__name__)

_PALETTE = px.colors.qualitative.Bold


# ── Type alias ───────────────────────────────────────────────────────────────
Partition = dict[str, int]


def _to_igraph(G: nx.Graph):
    """
    Convert NetworkX → igraph with correct edge-weight alignment.
    """
    import igraph as ig

    nodes = list(G.nodes())
    node_to_int = {n: i for i, n in enumerate(nodes)}

    g = ig.Graph(directed=G.is_directed())
    g.add_vertices(len(nodes))

    edges = []
    weights = []

    for u, v, d in G.edges(data=True):
        edges.append((node_to_int[u], node_to_int[v]))
        weights.append(float(d.get("weight", 1.0)))

    g.add_edges(edges)

    if len(weights) > 0:
        g.es["weight"] = weights

    return g, nodes


def _run_leiden(
    G: nx.Graph,
    seed: int = 42,
    resolution: float = 1.0,
    n_iterations: int = -1,
    partition_type: str = "RB",   # default: dimensionless modularity
) -> Partition:
    """
    Leiden optimisation for hybrid weighted earthquake network.

    partition_type:
        - "RB"  (default) – Reichardt-Bornholdt modularity; dimensionless
                resolution γ, weight-scale-invariant. Matches the standard
                "Louvain γ" parameter in the literature. Use this for any
                graph with non-trivial edge-weight scale (e.g. the hybrid
                network with weights spanning many decades).
        - "CPM" – Constant Potts Model; γ is an *absolute* density threshold
                in edge-weight units. Only sensible when edge weights are
                normalised to ~O(1); on a wide-range weight distribution the
                resolution becomes scale-dependent and γ ≈ 1 either over- or
                under-merges depending on the median weight.
    """
    import leidenalg

    g, nodes = _to_igraph(G)

    weights = "weight" if "weight" in g.es.attributes() else None

    if partition_type == "CPM":
        part = leidenalg.find_partition(
            g,
            leidenalg.CPMVertexPartition,
            weights=weights,
            seed=seed,
            resolution_parameter=resolution,
            n_iterations=n_iterations,
        )

    else:  # fallback to RB (your original)
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            seed=seed,
            resolution_parameter=resolution,
            n_iterations=n_iterations,
        )

    return {nodes[i]: cid for cid, members in enumerate(part) for i in members}


def run_louvain_hybrid(
    G: nx.Graph,
    seed: int = 42,
    resolution: float = 1.0,
    partition_type: str = "RB",
) -> Partition:
    """
    Community detection on undirected hybrid network.

    Uses Reichardt-Bornholdt modularity by default (``partition_type="RB"``)
    – dimensionless γ, weight-scale-invariant. Pass ``partition_type="CPM"``
    only if you have a normalised-weight graph and know that γ is a meaningful
    absolute density threshold for your data.
    """
    G_und = G.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))

    return _run_leiden(G_und, seed=seed, resolution=resolution,
                       partition_type=partition_type)


def run_louvain_directed_hybrid(
    G: nx.DiGraph,
    seed: int = 42,
    resolution: float = 1.0,
    partition_type: str = "RB",
) -> Partition:
    """
    Directed community detection for hybrid network. See
    :func:`run_louvain_hybrid` for the ``partition_type`` discussion.
    """
    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))

    return _run_leiden(G, seed=seed, resolution=resolution,
                       partition_type=partition_type)


def run_louvain_consensus_hybrid(
    G: nx.Graph,
    n_runs: int = 20,
    resolution: float = 1.0,
    directed: bool = False,
    threshold: float = 0.5,
    max_iter: int = 1,
    sample_pairs: bool = False,
    max_pairs_per_comm: int = 500,
) -> dict:
    """
    Consensus community detection (Lancichinetti & Fortunato 2012) for
    hybrid weighted network: run Louvain ``n_runs`` times → build the
    co-occurrence graph H where edge (u,v) has weight = fraction of runs
    in which u, v ended up in the same community → run Louvain once on H,
    keeping only co-occurrence edges with weight ≥ ``threshold``.

    Parameters
    ----------
    n_runs : int
        Number of Louvain runs to average over.
    resolution : float
        γ for each individual Louvain run. MUST match the γ used for any
        plain Louvain you are comparing against – different γ give partitions
        at different granularity scales (low NMI without method disagreement).
    threshold : float
        Lancichinetti-Fortunato cutoff. Keep co-occurrence edges with
        normalised weight ≥ threshold. 0.5 is standard.
    max_iter : int
        Number of iterative consensus rounds. **Default 1** – the standard
        algorithm is a single round (run on G, build H, run on H). Iterating
        replaces G with H repeatedly, which (a) loses directedness because H
        is undirected by construction and (b) compounds shrinkage, ending
        with many micro-components. Set >1 only if you have a specific reason.
    sample_pairs : bool
        Subsample members of large communities before counting co-occurrence.
        **Default False** – sampling drops genuine co-occurrences below the
        threshold for any community larger than ``max_pairs_per_comm`` (each
        node has p = max_pairs_per_comm/|C| of being sampled, so a pair has
        p² of both being in the same run's sample; for |C|=1000 and
        max=500 that's 25% per run, well below the 50% threshold even with
        perfect within-community stability).
    max_pairs_per_comm : int
        Only used when ``sample_pairs=True``; cap on members sampled per
        community per run.
    """

    import random
    from collections import defaultdict

    def run_partition(graph):
        partitions = []
        for seed in range(n_runs):

            if directed:
                p = run_louvain_directed_hybrid(
                    graph, seed=seed, resolution=resolution
                )
            else:
                p = run_louvain_hybrid(
                    graph, seed=seed, resolution=resolution
                )

            partitions.append(p)

        return partitions

    def build_consensus_graph(partitions):
        edge_weights = defaultdict(int)

        for part in partitions:

            comms = defaultdict(list)
            for n, c in part.items():
                comms[c].append(n)

            for members in comms.values():

                # subsample large communities
                if sample_pairs and len(members) > max_pairs_per_comm:
                    members = random.sample(members, max_pairs_per_comm)

                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        a, b = members[i], members[j]
                        if a > b:
                            a, b = b, a
                        edge_weights[(a, b)] += 1

        # build consensus graph
        H = nx.Graph()

        for (a, b), w in edge_weights.items():
            w = w / n_runs  # normalize to [0,1]

            if w >= threshold:
                H.add_edge(a, b, weight=w)

        return H

    current_graph = G.copy()

    for _ in range(max_iter):

        partitions = run_partition(current_graph)
        new_graph = build_consensus_graph(partitions)

        # stopping conditions
        if new_graph.number_of_edges() == 0:
            break

        if new_graph.number_of_edges() == current_graph.number_of_edges():
            break

        current_graph = new_graph

    # final partition
    if directed:
        return run_louvain_directed_hybrid(
            current_graph, resolution=resolution
        )
    else:
        return run_louvain_hybrid(
            current_graph, resolution=resolution
        )


def run_infomap_hybrid(
    G: nx.Graph,
    directed: bool = True,
    seed: int = 42,
) -> dict:
    """
    InfoMap community detection adapted for hybrid weighted earthquake network.

    Works with:
    - exponential interaction weights
    - directed or undirected graphs
    - sparse thresholded networks
    """

    try:
        from infomap import Infomap
    except ImportError:
        raise ImportError("pip install infomap")

    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))

    nodes = list(G.nodes())
    node_to_int = {n: i for i, n in enumerate(nodes)}

    # Pass directed/silent/seed via the constructor – the attribute-set form
    # (``im.directed = True`` after init) is non-standard and silently no-ops
    # on current infomap versions, which would run undirected flow on a
    # directed graph (different community boundaries entirely).
    im = Infomap(directed=directed, silent=True, seed=seed)

    # Add weighted edges (hybrid interaction strengths)
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))

        # safety: Infomap expects strictly positive weights
        if w <= 0:
            continue

        im.add_link(node_to_int[u], node_to_int[v], w)

    im.run()

    partition = {}

    # robust extraction (API-safe)
    for node in im.tree:
        if node.is_leaf:
            partition[nodes[node.node_id]] = node.module_id

    return partition


# ====================================================================================
# Density-based geographic clustering (spatial null baseline)
# ====================================================================================


def run_hdbscan_geo_hybrid(
    G: nx.Graph,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    target_crs: str = "epsg:32632",
) -> dict:
    """
    HDBSCAN on the geographic coordinates of the cells – a *spatial null
    baseline* for community detection. Ignores the network entirely and
    clusters cells purely by (projected) (x, y) position.

    Useful as the "is the network adding signal beyond spatial proximity?"
    contrast against the graph-aware methods (Louvain, InfoMap, MM-SBM):
    high NMI with HDBSCAN-geo means the network methods are mostly
    rediscovering geography; moderate NMI means the network captures
    structure beyond spatial clustering.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Network. Each node must have ``lat`` and ``lon`` attributes (assigned
        by ``_assign_node_coords`` in ``src/network_custom.py``).
    min_cluster_size : int, default 10
        HDBSCAN minimum cluster size. Matches the ``min_community_size=10``
        filter convention used elsewhere in the project.
    min_samples : int or None
        HDBSCAN ``min_samples``. ``None`` defaults to ``min_cluster_size`` –
        a relatively conservative setting that yields fewer noise points.
    target_crs : str, default ``"epsg:32632"``
        Metric CRS used to project lat/lon into kilometres before clustering.
        Italy: UTM Zone 32N. Mirrors the project's standard projection.

    Returns
    -------
    partition : dict
        ``{node_id: cluster_id}``. HDBSCAN noise points (label ``-1``) are
        re-labelled as unique singleton clusters (one cluster id per noise
        node), so the ``≥ 10`` cell filter applied downstream cleanly drops
        them from NMI without lumping them into a single artificial
        "noise community".

    References
    ----------
    Campello, R.J.G.B., Moulavi, D. & Sander, J. (2013).
    *Density-Based Clustering Based on Hierarchical Density Estimates.*
    PAKDD.

    McInnes, L. & Healy, J. (2017). *Accelerated Hierarchical Density
    Based Clustering.* IEEE ICDMW.
    """
    import hdbscan
    from pyproj import Transformer

    nodes = list(G.nodes())
    coords_ll = np.asarray(
        [(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in nodes],
        dtype=np.float64,
    )

    # Project (lat, lon) → metric (x, y) in km so distances are physical.
    fwd = Transformer.from_crs("epsg:4326", target_crs, always_xy=True)
    xs_m, ys_m = fwd.transform(coords_ll[:, 1], coords_ll[:, 0])
    xy_km = np.column_stack([xs_m, ys_m]) / 1000.0

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples if min_samples is not None else min_cluster_size,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(xy_km)

    # Re-label noise (-1) as unique singletons so the ≥ 10-cell filter excludes
    # them rather than treating all noise as one giant "community".
    next_singleton = int(labels.max()) + 1 if (labels >= 0).any() else 0
    partition: dict = {}
    for n, lbl in zip(nodes, labels):
        if int(lbl) == -1:
            partition[n] = next_singleton
            next_singleton += 1
        else:
            partition[n] = int(lbl)
    return partition


# ====================================================================================

# ── Geographical community map ────────────────────────────────────────────────

# ====================================================================================


def plot_community_geo_hybrid(
    G: nx.Graph,
    community_map: Partition,
    title: str = "",
    center_lat: float = 41.9,
    center_lon: float = 12.5,
    zoom: float = 5,
    bounds: dict | None = None,
    min_community_size: int = 50,
    method_name: str = "",
    height: int = 600,
    width: int = 1100,
    save: bool = True,
    renderer: str | None = None,
) -> None:
    """
    Geographic visualization of communities in the hybrid earthquake network.

    Node size reflects weighted degree (interaction strength).
    Only sufficiently large communities are shown.

    Parameters
    ----------
    renderer : str or None
        Plotly renderer passed to ``fig.show``. ``None`` (default) shows the live
        interactive WebGL map. Pass ``"png"`` (or ``"svg"`` / ``"pdf"``) to render a
        **static image** instead, so repeated maps do not exhaust the browser's
        WebGL context cap (older maps going blank).
    """

    rows = []
    for n in G.nodes():
        if "lat" not in G.nodes[n]:
            continue

        rows.append({
            "cell_id":   n,
            "community": str(community_map.get(n, -1)),
            "lat":       G.nodes[n]["lat"],
            "lon":       G.nodes[n]["lon"],
            "strength":  G.degree(n, weight="weight"),  # ← FIXED
        })

    df = pd.DataFrame(rows)
    df = df[df["community"] != "-1"]   # to remove nodes excluded from consensus louvain

    # filter small communities
    counts = df["community"].value_counts()
    large = counts[counts >= min_community_size].index
    df = df[df["community"].isin(large)].copy()

    n_shown = df["community"].nunique()

    if len(df) == 0:
        print("No communities large enough to display.")
        return

    # log-scale size (robust for hybrid weights)
    df["size_val"] = np.log1p(df["strength"]).clip(lower=0.5)

    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color="community",
        size="size_val",
        size_max=18,
        color_discrete_sequence=_PALETTE,
        hover_name="community",
        hover_data={
            "lat": ":.3f",
            "lon": ":.3f",
            "strength": ":.3e",   # scientific notation (important!)
            "size_val": False,
        },
        mapbox_style="carto-positron",
        title=pres_title(
            f"Seismic Communities: {method_name}",
            f"{n_shown} communities (size ≥ {min_community_size} cells), {title}"
            if title else f"{n_shown} communities (size ≥ {min_community_size} cells)",
        ),
    )

    fig.update_traces(marker=dict(opacity=0.7))

    map_cfg = {
        "center": {"lat": center_lat, "lon": center_lon},
        "zoom": zoom,
    }

    if bounds is not None:
        map_cfg["bounds"] = bounds

    fig.update_layout(
        mapbox=map_cfg,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        width=width,
        height=height,
        showlegend=True,
    )

    if save:
        save_plotly(fig, f"community_geo_hybrid_{_slug(method_name)}_{_slug(title)}")

    fig.show(renderer)


# ====================================================================================
# Communities vs DISS seismogenic sources (fault validation)
# ====================================================================================

def load_diss_faults(
    diss_dir,
    italy_only: bool = True,
    with_iss: bool = True,
):
    """
    Load DISS 3.3.1 seismogenic-source layers as GeoDataFrames in EPSG:4326.

    Parameters
    ----------
    diss_dir : str or Path
        Directory containing ``csspln331.geojson`` (Composite Seismogenic
        Sources), ``iss331.geojson`` (Individual Seismogenic Sources) and,
        optionally, ``limits_IT_regions.geojson`` (Italian regional boundaries
        used as an offline basemap reference).
    italy_only : bool
        Keep only sources whose ``idsource`` code starts with ``IT``.
    with_iss : bool
        Also load the Individual Seismogenic Sources layer.

    Returns
    -------
    dict
        ``{"css": GeoDataFrame|None, "iss": GeoDataFrame|None,
           "regions": GeoDataFrame|None}`` – all in EPSG:4326 (lon/lat),
        directly compatible with the lon/lat community maps.
    """
    import geopandas as gpd  # heavy optional dependency – import lazily

    diss_dir = Path(diss_dir)

    def _load(name: str):
        p = diss_dir / name
        if not p.exists():
            log.warning("DISS layer not found: %s", p)
            return None
        gdf = gpd.read_file(p)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        return gdf.to_crs(4326)

    css     = _load("csspln331.geojson")
    iss     = _load("iss331.geojson") if with_iss else None
    regions = _load("limits_IT_regions.geojson")

    if italy_only:
        for key, gdf in (("css", css), ("iss", iss)):
            if gdf is not None and "idsource" in gdf.columns:
                gdf = gdf[gdf["idsource"].astype(str).str.startswith("IT")]
                if key == "css":
                    css = gdf
                else:
                    iss = gdf

    return {"css": css, "iss": iss, "regions": regions}


def _community_points_df(
    G: nx.Graph,
    community_map: Partition,
    min_community_size: int,
) -> pd.DataFrame:
    """Community nodes as a lon/lat point DataFrame (shared helper)."""
    rows = []
    for n in G.nodes():
        if "lat" not in G.nodes[n]:
            continue
        rows.append({
            "cell_id":   n,
            "community": str(community_map.get(n, -1)),
            "lat":       G.nodes[n]["lat"],
            "lon":       G.nodes[n]["lon"],
            "strength":  G.degree(n, weight="weight"),
        })
    df = pd.DataFrame(rows)
    df = df[df["community"] != "-1"]
    counts = df["community"].value_counts()
    large = counts[counts >= min_community_size].index
    return df[df["community"].isin(large)].copy()


def _geom_to_lonlat_lines(gdf) -> tuple[list, list]:
    """
    Flatten a (Multi)Polygon / (Multi)LineString GeoDataFrame to ``lon``/``lat``
    arrays with ``None`` separators between disjoint segments.

    The ``None`` gaps let a single Plotly ``Scattermap`` line trace draw every
    fault as one trace (so it renders *above* the community markers, which the
    legacy ``mapbox.layers`` path could not). Polygon exteriors are traced as
    closed rings.
    """
    lons: list = []
    lats: list = []

    def _ring(coords) -> None:
        for x, y in coords:
            lons.append(x)
            lats.append(y)
        lons.append(None)
        lats.append(None)

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        gt = geom.geom_type
        if gt == "Polygon":
            _ring(geom.exterior.coords)
        elif gt == "MultiPolygon":
            for part in geom.geoms:
                _ring(part.exterior.coords)
        elif gt == "LineString":
            _ring(geom.coords)
        elif gt == "MultiLineString":
            for part in geom.geoms:
                _ring(part.coords)
    return lons, lats


# Token-free Carto raster basemaps without place labels (cleaner for slides).
# Selected by passing the key as ``basemap_style``; rendered as a raster layer
# beneath the data traces (Plotly has no built-in "nolabels" mapbox style).
_NOLABELS_TILES = {
    "carto-positron-nolabels":
        "https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
    "carto-darkmatter-nolabels":
        "https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png",
}


def _link_traces_mapbox(
    H,
    pos: dict,
    weight: str = "weight",
    top_frac: float | None = 0.1,
    n_bins: int = 3,
    base_color: tuple = (58, 67, 80),
    alpha_range: tuple = (0.12, 0.45),
    width_range: tuple = (0.4, 2.2),
):
    """
    Weight-banded ``go.Scattermapbox`` line traces for a graph's edges, for
    drawing the interaction links on a tile basemap. Edges are bucketed into
    ``n_bins`` weight bands (Plotly mapbox takes one line width per trace);
    ``top_frac`` keeps only the strongest fraction (the backbone). ``base_color``
    (RGB), ``alpha_range`` and ``width_range`` set the min→max opacity / line
    width across the bands – raise them for crisp links drawn *on top* of faded
    markers. Shared by the interactive geo-edges map and the community/fault
    overlay. Returns a list of ``go.Scattermapbox`` (possibly empty).
    """
    edges = [(u, v, float(d.get(weight, 1.0))) for u, v, d in H.edges(data=True)]
    if not edges:
        return []
    basis = np.log1p(np.array([w for _, _, w in edges], dtype=float))
    if top_frac is not None and 0.0 < top_frac < 1.0:
        keep = basis >= np.quantile(basis, 1.0 - top_frac)
        edges = [e for e, k in zip(edges, keep) if k]
        basis = basis[keep]
    if len(edges) == 0:
        return []
    order = np.argsort(basis)
    r, g, b = base_color
    a0, a1 = alpha_range
    w0, w1 = width_range
    traces = []
    for b_i, bin_idx in enumerate(np.array_split(order, n_bins)):
        if len(bin_idx) == 0:
            continue
        lons: list = []
        lats: list = []
        for j in bin_idx:
            u, v, _ = edges[j]
            lons += [pos[u][0], pos[v][0], None]
            lats += [pos[u][1], pos[v][1], None]
        frac = (b_i + 1) / n_bins
        traces.append(go.Scattermapbox(
            lon=lons, lat=lats, mode="lines",
            line=dict(width=w0 + (w1 - w0) * frac, color=f"rgb({r},{g},{b})"),
            opacity=a0 + (a1 - a0) * frac,
            hoverinfo="skip", showlegend=False))
    return traces


def plot_communities_faults_overlay_hybrid(
    G: nx.Graph,
    community_map: Partition,
    diss_dir,
    title: str = "",
    method_name: str = "",
    center_lat: float = 41.9,
    center_lon: float = 12.5,
    zoom: float = 0,
    bounds: dict | None = None,
    min_community_size: int = 10,
    height: int = 700,
    width: int = 770,
    italy_only: bool = True,
    with_iss: bool = False,
    basemap_style: str = "carto-positron-nolabels",
    fault_color: str = "#222222",
    fault_casing_color: str = "rgba(255,255,255,0.9)",
    marker_opacity: float = 0.75,
    size_max: int = 11,
    draw_faults: bool = True,
    draw_links: bool = False,
    link_top_frac: float = 0.1,
    link_weight: str = "weight",
    links_on_top: bool = False,
    link_color: tuple = (58, 67, 80),
    link_alpha: tuple = (0.12, 0.45),
    link_width: tuple = (0.4, 2.2),
    save: bool = True,
    renderer: str | None = None,
) -> None:
    """
    Interactive overlay of detected communities and DISS faults on one map (A).

    Community cells are coloured by community (marker size = weighted degree).
    DISS seismogenic sources are drawn as **line traces on top of** the markers
    (Plotly ``mapbox.layers`` always render *beneath* data traces, which buried
    the faults under the dense, on-fault community markers). Each fault is drawn
    twice – a wider light casing then a thin dark core – so it stays legible
    over both the basemap and dark community markers. Because the community
    palette (``Bold``) already spans the full hue wheel, the fault colour is kept
    achromatic by default (near-black + white casing) so it never reads as
    "another community".

    Parameters
    ----------
    basemap_style : str
        Basemap. The default ``"carto-positron-nolabels"`` (and
        ``"carto-darkmatter-nolabels"``) are token-free Carto raster basemaps
        without place labels, which keeps the slide clean; any built-in Plotly
        mapbox style name (e.g. ``"carto-positron"``) is also accepted. The light
        basemaps pair with the default dark-grey faults; for a dark basemap pass a
        bright ``fault_color`` such as ``"#00e5ff"`` (cyan is the one vivid hue
        absent from ``Bold``) with ``fault_casing_color="rgba(0,0,0,0.6)"``.
    fault_color, fault_casing_color : str
        Core and casing (halo) colours for the fault line traces.
    marker_opacity : float
        Community-marker opacity; lower lets the faults show through dense areas.
    size_max : int
        Maximum community-marker size (weighted-degree scaled).
    with_iss : bool
        Also overlay Individual Seismogenic Sources. Off by default – at this
        zoom the ISS planes add clutter and the CSS outlines carry the structure.
    draw_faults : bool
        Draw the DISS fault traces. Set ``False`` for a links-only community map
        on the same basemap (no faults).
    draw_links : bool
        Also draw the **interaction links** (top-``link_top_frac`` by weight,
        beneath the markers). Combine with ``draw_faults`` for a communities +
        faults + links map, or with ``draw_faults=False`` for links only.
    link_top_frac : float
        Fraction of strongest edges to draw as the link backbone.
    link_weight : str
        Edge-weight attribute key used to rank and scale the links.
    links_on_top : bool
        Draw the links *above* the markers instead of beneath. Pair with a low
        ``marker_opacity`` and a crisp ``link_color``/``link_alpha`` to fade the
        community field into context and make the backbone the focus.
    link_color : tuple
        Link RGB colour.
    link_alpha, link_width : tuple
        (min, max) opacity / line width across the weight bands.
    renderer : str or None
        Plotly renderer passed to ``fig.show``. ``None`` (default) shows the live
        interactive WebGL map. Pass ``"png"`` (or ``"svg"`` / ``"pdf"``) to render a
        **static image** instead, so repeated maps do not exhaust the browser's
        WebGL context cap (older maps going blank).
    """
    faults = load_diss_faults(diss_dir, italy_only=italy_only, with_iss=with_iss) \
        if draw_faults else {}

    df = _community_points_df(G, community_map, min_community_size)
    if df.empty:
        print("No communities large enough to display.")
        return
    n_shown = df["community"].nunique()
    df["size_val"] = np.log1p(df["strength"]).clip(lower=0.5)

    # Two-line title (main + smaller subtitle) so it never overflows the width.
    _what = ("Communities, faults & links" if (draw_faults and draw_links)
             else "Communities vs DISS faults" if draw_faults
             else "Community network links" if draw_links
             else "Communities")
    title_text = pres_title(
        f"{_what}: {method_name}",
        f"{n_shown} communities (size ≥ {min_community_size} cells), {title}"
        if title else f"{n_shown} communities (size ≥ {min_community_size} cells)",
    )

    # No-labels basemaps are raster tiles drawn under the traces; Plotly's own
    # mapbox style is set to "white-bg" in that case.
    raster_url = _NOLABELS_TILES.get(basemap_style)
    px_style = "white-bg" if raster_url else basemap_style

    fig = px.scatter_mapbox(
        df, lat="lat", lon="lon",
        color="community", size="size_val", size_max=size_max,
        color_discrete_sequence=_PALETTE,
        hover_name="community",
        hover_data={"lat": ":.3f", "lon": ":.3f",
                    "strength": ":.3e", "size_val": False},
        mapbox_style=px_style,
        title=title_text,
    )
    fig.update_traces(marker=dict(opacity=marker_opacity))

    # Interaction links. Beneath the markers by default (prepended so the cells
    # stay on top); with ``links_on_top`` they are drawn above a faded marker
    # field so the backbone is the focus. Faults (added below) stay on top of all.
    link_traces = []
    if draw_links:
        shown = set(df["cell_id"])
        H = G.subgraph(n for n in shown if n in G)
        pos = {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in H.nodes()}
        link_traces = _link_traces_mapbox(
            H, pos, weight=link_weight, top_frac=link_top_frac,
            base_color=link_color, alpha_range=link_alpha, width_range=link_width)
        if link_traces and not links_on_top:
            for t in link_traces:
                fig.add_trace(t)
            # Reorder the just-added links to the front (a valid permutation of
            # existing traces) so they render beneath the markers.
            d = list(fig.data)
            n_link = len(link_traces)
            fig.data = tuple(d[-n_link:] + d[:-n_link])

    # Fault sources as line traces ON TOP of the markers (casing + dark core).
    # Each source layer gets one casing trace and one core trace so disjoint
    # segments share a single legend entry.
    if draw_faults:
        for key, core_w, case_w in (("css", 1.2, 3.0), ("iss", 0.9, 2.4)):
            layer = faults.get(key)
            if layer is None or layer.empty:
                continue
            lons, lats = _geom_to_lonlat_lines(layer)
            if not lons:
                continue
            fig.add_trace(go.Scattermapbox(
                lon=lons, lat=lats, mode="lines",
                line=dict(color=fault_casing_color, width=case_w),
                hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scattermapbox(
                lon=lons, lat=lats, mode="lines",
                line=dict(color=fault_color, width=core_w),
                name="DISS faults" if key == "css" else "DISS ISS",
                hoverinfo="skip",
                showlegend=(key == "css")))

    # Links drawn ON TOP of the (faded) markers, after faults so faults stay above.
    if draw_links and links_on_top and link_traces:
        for t in link_traces:
            fig.add_trace(t)

    map_cfg = {"center": {"lat": center_lat, "lon": center_lon}, "zoom": zoom}
    if bounds is not None:
        map_cfg["bounds"] = bounds
    if raster_url:
        map_cfg["layers"] = [{
            "below": "traces", "sourcetype": "raster",
            "source": [raster_url],
            "sourceattribution": "© CARTO © OpenStreetMap contributors",
        }]

    fig.update_layout(mapbox=map_cfg,
                      margin={"r": 0, "t": 55, "l": 0, "b": 0},
                      title=dict(x=0.45, xanchor="center"),
                      width=width, height=height, showlegend=True)

    if save:
        variant = ("_faults_links" if (draw_faults and draw_links)
                   else "_links" if draw_links
                   else "")
        save_plotly(fig,
                    f"communities_vs_faults_overlay{variant}_{_slug(method_name)}_{_slug(title)}")
    fig.show(renderer)


def plot_network_overview_hybrid(
    G: nx.Graph,
    diss_dir=None,
    title: str = "",
    center_lat: float = 41.9,
    center_lon: float = 12.5,
    zoom: float = 0,
    bounds: dict | None = None,
    weight: str = "weight",
    color_by: str = "strength",
    node_top_frac: float | None = 0.30,
    link_top_frac: float = 0.02,
    size_range: tuple = (5.0, 16.0),
    link_color: tuple = (40, 45, 55),
    link_alpha: tuple = (0.22, 0.62),
    draw_faults: bool = False,
    giant_only: bool = True,
    clip_pct: tuple = (2.0, 98.0),
    italy_only: bool = True,
    with_iss: bool = False,
    basemap_style: str = "carto-positron-nolabels",
    fault_color: str = "#222222",
    fault_casing_color: str = "rgba(255,255,255,0.9)",
    colorscale: str = "plasma",
    height: int = 700,
    width: int = 770,
    save: bool = True,
    renderer: str | None = None,
) -> None:
    """
    Network **skeleton** overview (post-construction, pre-community-detection):
    only the **top cells** (top ``node_top_frac`` by the chosen metric) are drawn,
    coloured and sized by that metric (``color_by="strength"`` → log₁₀ strength,
    the default; ``color_by="degree"`` → raw degree, linear) on a sequential
    ``plasma`` scale, with only the **strongest links** (top ``link_top_frac``)
    among them as the interaction backbone, on the ``carto-positron-nolabels`` basemap.
    Showing every one of the ~3 k cells turns the 20 km grid into a dense dot
    field and the links into texture; thresholding to the strong hubs + a sparse
    backbone makes the structure read as a network. The "establishing shot" of
    what was built, before any partition colours the cells.

    Set ``draw_faults=True`` (and pass ``diss_dir``) to overlay the DISS
    seismogenic sources; otherwise it is a faults-free structural overview.

    Parameters
    ----------
    G : nx.Graph
        Hybrid network with ``lon``/``lat`` node attributes and weighted edges.
    node_top_frac : float or None
        Keep only this fraction of the top cells (``0.30`` = top 30 %).
        ``None`` keeps every cell (the dense-grid view).
    color_by : str
        Node metric for colour, size and thresholding: ``"strength"`` (weighted
        degree, shown as log₁₀; default) or ``"degree"`` (raw count, linear).
    link_top_frac : float
        Fraction of the strongest edges *among the kept cells* drawn as the
        backbone.
    size_range : tuple
        (min, max) marker size mapped onto log-strength.
    link_color, link_alpha : tuple
        Link RGB colour and (min, max) opacity across the weight bands.
    draw_faults : bool
        Overlay DISS faults (needs ``diss_dir``). Off by default.
    giant_only : bool
        Restrict to the largest (weakly-)connected component before thresholding.
    clip_pct : tuple
        (low, high) percentiles of log₁₀(strength) used as the colour limits.
    colorscale : str
        Sequential colourscale for log-strength (project convention: ``plasma``).
    renderer : str or None
        Plotly renderer passed to ``fig.show``. ``None`` (default) shows the live
        interactive WebGL map. Pass ``"png"`` (or ``"svg"`` / ``"pdf"``) to render a
        **static image** instead: static images do not hold a WebGL context, so the
        same map can be shown repeatedly without older maps going blank when the
        browser hits its WebGL context cap.
    """
    if giant_only and G.number_of_nodes():
        comps = (nx.weakly_connected_components(G) if G.is_directed()
                 else nx.connected_components(G))
        G = G.subgraph(max(comps, key=len))

    nodes = [n for n in G.nodes() if "lat" in G.nodes[n]]
    if not nodes:
        print("No georeferenced nodes to display.")
        return

    by_degree = color_by.lower() == "degree"
    if by_degree:
        vals = np.array([float(G.degree(n)) for n in nodes], dtype=float)          # unweighted count
    else:
        vals = np.array([float(G.degree(n, weight=weight)) for n in nodes], dtype=float)  # strength

    # keep only the top cells (the hubs) so the map is a skeleton, not a grid
    if node_top_frac is not None and 0.0 < node_top_frac < 1.0:
        thr = np.quantile(vals, 1.0 - node_top_frac)
        keep = vals >= thr
        nodes = [n for n, k in zip(nodes, keep) if k]
        vals = vals[keep]
    if not nodes:
        print("No cells left after thresholding.")
        return

    if by_degree:
        # degree: linear colour + linear sizing. Use the TRUE min/max (no percentile
        # clip): degree is a plain linear count, so the colorbar should match the
        # standalone hub map. (clip_pct is only needed for strength's ~6-decade range.)
        color_vals = vals
        cmin, cmax = (float(vals.min()), float(vals.max())) if vals.size else (None, None)
        cbar_title = "Degree"
        s_base = vals
    else:
        # strength: log10 colour + log1p sizing (weights span several decades)
        color_vals = np.log10(np.clip(vals, 1e-12, None))
        finite = color_vals[np.isfinite(color_vals)]
        cmin, cmax = (np.percentile(finite, clip_pct) if finite.size else (None, None))
        cbar_title = "log<sub>10</sub>(strength)"
        s_base = np.log1p(vals)

    lo, hi = float(s_base.min()), float(s_base.max())
    norm = (s_base - lo) / (hi - lo) if hi > lo else np.full_like(s_base, 0.5)
    sizes = size_range[0] + (size_range[1] - size_range[0]) * norm

    lon = np.array([G.nodes[n]["lon"] for n in nodes], dtype=float)
    lat = np.array([G.nodes[n]["lat"] for n in nodes], dtype=float)

    raster_url = _NOLABELS_TILES.get(basemap_style)
    px_style = "white-bg" if raster_url else basemap_style

    fig = go.Figure()

    # backbone links (strongest among the kept hubs) beneath the nodes
    pos = {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in nodes}
    H = G.subgraph(nodes)
    for t in _link_traces_mapbox(H, pos, weight=weight, top_frac=link_top_frac,
                                 base_color=link_color, alpha_range=link_alpha):
        fig.add_trace(t)

    # hub nodes coloured by the chosen metric
    hov = ([f"cell {n}<br>degree {int(v)}" for n, v in zip(nodes, vals)] if by_degree
           else [f"cell {n}<br>strength {v:.3e}" for n, v in zip(nodes, vals)])
    fig.add_trace(go.Scattermapbox(
        lon=lon, lat=lat, mode="markers",
        marker=dict(size=sizes, color=color_vals, colorscale=colorscale,
                    cmin=cmin, cmax=cmax, opacity=0.9,
                    colorbar=dict(title=cbar_title)),
        text=hov,
        hoverinfo="text", showlegend=False,
    ))

    # optional DISS faults on top (casing + dark core)
    if draw_faults and diss_dir is not None:
        faults = load_diss_faults(diss_dir, italy_only=italy_only, with_iss=with_iss)
        for key, core_w, case_w in (("css", 1.2, 3.0), ("iss", 0.9, 2.4)):
            layer = faults.get(key)
            if layer is None or layer.empty:
                continue
            lons, lats = _geom_to_lonlat_lines(layer)
            if not lons:
                continue
            fig.add_trace(go.Scattermapbox(
                lon=lons, lat=lats, mode="lines",
                line=dict(color=fault_casing_color, width=case_w),
                hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scattermapbox(
                lon=lons, lat=lats, mode="lines",
                line=dict(color=fault_color, width=core_w),
                name="DISS faults" if key == "css" else "DISS ISS",
                hoverinfo="skip", showlegend=(key == "css")))

    link_pct = int(round(link_top_frac * 100))
    node_pct = (f"top {int(round(node_top_frac * 100))}%"
                if node_top_frac is not None and 0.0 < node_top_frac < 1.0
                else "all")
    colour_short = "degree" if by_degree else "strength"
    colour_lbl = "degree" if by_degree else "log<sub>10</sub>(weighted strength)"
    _what = (f"Network overview by {colour_short} "
             f"({node_pct} nodes, top {link_pct}% links)")
    if draw_faults:
        _what += " + DISS faults"
    title_text = pres_title(
        f"{_what} – {title}" if title else _what,
        f"colour = {colour_lbl}, {len(nodes):,} cells shown",
    )

    map_cfg = {"style": px_style,
               "center": {"lat": center_lat, "lon": center_lon}, "zoom": zoom}
    if bounds is not None:
        map_cfg["bounds"] = bounds
    if raster_url:
        map_cfg["layers"] = [{
            "below": "traces", "sourcetype": "raster",
            "source": [raster_url],
            "sourceattribution": "© CARTO © OpenStreetMap contributors",
        }]

    fig.update_layout(mapbox=map_cfg,
                      margin={"r": 0, "t": 55, "l": 0, "b": 0},
                      title=dict(text=title_text, x=0.45, xanchor="center"),
                      # legend (DISS faults key) bottom-left, clear of the right colorbar
                      legend=dict(x=0.01, y=0.01, xanchor="left", yanchor="bottom",
                                  bgcolor="rgba(255,255,255,0.7)",
                                  bordercolor="#cccccc", borderwidth=1),
                      width=width, height=height, showlegend=draw_faults)

    if save:
        variant = "_faults" if draw_faults else ""
        save_plotly(fig, f"network_overview{variant}_{_slug(title)}")
    fig.show(renderer)


# ====================================================================================

# ──────────────────────────────── NMI ────────────────────────────────────────────────

# ====================================================================================


def align_partitions(part1: dict, part2: dict):
    """
    Align two partitions on common nodes (stable ordering).
    """
    common_nodes = sorted(set(part1.keys()) & set(part2.keys()))

    labels1 = np.array([part1[n] for n in common_nodes])
    labels2 = np.array([part2[n] for n in common_nodes])

    return labels1, labels2


def compute_nmi(part1: dict, part2: dict) -> float:
    labels1, labels2 = align_partitions(part1, part2)
    return normalized_mutual_info_score(labels1, labels2)


def compute_nmi_matrix(partitions: dict) -> pd.DataFrame:
    """
    Compute symmetric NMI similarity matrix between methods.
    """
    methods = list(partitions.keys())
    n = len(methods)

    mat = np.zeros((n, n))

    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            nmi = compute_nmi(partitions[methods[i]], partitions[methods[j]])
            mat[i, j] = nmi
            mat[j, i] = nmi

    return pd.DataFrame(mat, index=methods, columns=methods)


def plot_nmi_heatmap(nmi_df: pd.DataFrame, tag: str = "", save: bool = True):
    plt.figure(figsize=(7, 6))

    sns.heatmap(
        nmi_df,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="YlGn",
        square=True,
        linewidths=0.5
    )

    plt.title("NMI between community methods")
    plt.tight_layout()
    if save:
        savefig(f"community_nmi_matrix_{tag}")
    plt.show()


# ====================================================================================
# Partition quality: the four course measures (Modularity, Ncut, InfoMap, NMI)
# ====================================================================================

def _map_equation_codelength_infomap(
    G: nx.Graph,
    partition: Partition,
    weight: str = "weight",
    directed: bool = True,
    seed: int = 42,
) -> float:
    """
    Two-level map-equation codelength (bits) of ``partition`` under InfoMap's flow.

    Rather than reconstructing the random-walk flow by hand – which mixes the
    teleporting PageRank visit rates with the raw non-teleporting transition
    matrix, so the visit distribution is not stationary under the operator used
    for the flow and the codelength does not match the objective InfoMap
    minimises – this hands the *fixed* partition to InfoMap and reads back the
    codelength it assigns under one self-consistent flow model. ``run`` is called
    with ``no_infomap=True`` so InfoMap only encodes the given clustering and
    performs no community optimisation. Self-loops are dropped and directedness
    is matched to :func:`run_infomap_hybrid` so every partition is scored on the
    same flow model as the InfoMap-detected one.

    Parameters
    ----------
    G : nx.Graph
        Weighted network the partition was computed on.
    partition : dict
        ``{node: community_id}``. Nodes absent from the partition are ignored.
    weight : str
        Edge-weight attribute.
    directed : bool
        Directed flow model (match the value used when running InfoMap).
    seed : int
        InfoMap seed.

    Returns
    -------
    float
        Codelength in bits, or ``nan`` if the partition covers no linked nodes.

    References
    ----------
    Rosvall & Bergstrom 2008, *PNAS* 105(4):1118-1123.
    """
    try:
        from infomap import Infomap
    except ImportError:
        raise ImportError("pip install infomap")

    nodes = [n for n in G.nodes() if n in partition]
    if not nodes:
        return float("nan")
    node_set = set(nodes)
    node_to_int = {n: i for i, n in enumerate(nodes)}

    im = Infomap(directed=directed, silent=True, seed=seed)
    n_links = 0
    for u, v, data in G.edges(data=True):
        if u == v or u not in node_set or v not in node_set:
            continue
        w = float(data.get(weight, 1.0))
        if w <= 0:
            continue
        im.add_link(node_to_int[u], node_to_int[v], w)
        n_links += 1
    if n_links == 0:
        return float("nan")

    # contiguous integer module ids for the fixed clustering
    comm_to_int = {c: i for i, c in enumerate(sorted({partition[n] for n in nodes}))}
    im.initial_partition = {node_to_int[n]: comm_to_int[partition[n]] for n in nodes}
    im.run(no_infomap=True)
    return float(im.codelength)


def score_partition_hybrid(
    G: nx.DiGraph,
    partition: Partition,
    pagerank: dict | None = None,
    weight: str = "weight",
    alpha: float = 0.85,
) -> dict:
    """
    Flow-based partition quality in the course's unified P_cc = C·P_nn·Cᵀ framework.

    Three intrinsic quality measures are derived from the random-walk flow induced
    by the directed weighted network (Rosvall & Bergstrom 2008; course notes). With
    P_cc[a,b] the probability the walker is in community a and steps to community b,
    p_a = Σ_b P_cc[a,b] the stationary probability of community a (p = C·r, r the
    PageRank visit probabilities), q_a = p_a − P_aa the exit probability of a, and
    z_a = {r_i : i ∈ a} the node visit probabilities inside a:

      * Modularity     Q    = Σ_a (P_aa − p_a²)              (maximize)
      * Normalized cut Ncut = 1 − (1/K) Σ_a P_aa / p_a       (minimize)
      * Map equation   L    = two-level codelength, bits     (minimize)

    Modularity and Ncut use the flow above (PageRank visit rates r + community
    aggregation). The map-equation codelength, however, is evaluated by InfoMap
    itself (:func:`_map_equation_codelength_infomap`), not reconstructed here.
    The hand-rolled walk mixes teleporting PageRank visit rates with a raw
    non-teleporting transition matrix, so r is not stationary under the operator
    used for the within-community flow (‖r·P − r‖₁ > 0); the resulting codelength
    does not match the objective InfoMap actually minimises and can rank a
    balanced partition above InfoMap's own solution on directed, chain-like,
    dangling-heavy networks (e.g. Abe-Suzuki). Delegating to InfoMap's encoder
    puts every partition on one self-consistent flow model, so the codelength is
    comparable across partitions: InfoMap (a greedy stochastic heuristic) sits at
    or near the minimum, and any partition that beats it genuinely compresses the
    flow better rather than winning on a model artifact.

    The fourth course quality measure, NMI, is pairwise between partitions and is
    reported separately via :func:`compute_nmi_matrix`.

    Parameters
    ----------
    G : nx.DiGraph
        Directed weighted network the partition was computed on.
    partition : dict
        ``{node: community_id}``. Nodes absent from the partition are ignored.
    pagerank : dict, optional
        Precomputed PageRank ``{node: r}``; computed once on ``G`` if omitted
        (pass it in when scoring several partitions on the same graph).
    weight : str
        Edge-weight attribute used for transitions and PageRank.
    alpha : float
        PageRank damping factor.

    Returns
    -------
    dict
        ``{"modularity", "ncut", "codelength", "n_communities"}``.
    """
    nodes = [n for n in G.nodes() if n in partition]
    if not nodes:
        return {"modularity": np.nan, "ncut": np.nan,
                "codelength": np.nan, "n_communities": 0}

    if pagerank is None:
        pagerank = nx.pagerank(G, alpha=alpha, weight=weight)

    # node visit probabilities r, restricted to scored nodes and renormalised
    r = {n: float(pagerank.get(n, 0.0)) for n in nodes}
    rs = sum(r.values())
    if rs <= 0:
        return {"modularity": np.nan, "ncut": np.nan,
                "codelength": np.nan, "n_communities": 0}
    r = {n: v / rs for n, v in r.items()}

    comms = sorted({partition[n] for n in nodes})
    idx = {c: i for i, c in enumerate(comms)}
    K = len(comms)
    node_set = set(nodes)

    # community stationary probability p_a = Σ_{i in a} r_i
    p = np.zeros(K)
    for n in nodes:
        p[idx[partition[n]]] += r[n]

    # out-strength per scored node (within the scored subgraph)
    s_out = {n: 0.0 for n in nodes}
    for i, j, w in G.edges(data=weight, default=1.0):
        if i in node_set and j in node_set:
            s_out[i] += float(w)

    # within-community flow P_aa = Σ_{i,j in a} r_i · w_ij / s_out_i
    P_within = np.zeros(K)
    for i, j, w in G.edges(data=weight, default=1.0):
        if i not in node_set or j not in node_set:
            continue
        so = s_out[i]
        if so <= 0:
            continue
        if partition[i] == partition[j]:
            P_within[idx[partition[i]]] += r[i] * float(w) / so
    # dangling nodes (no out-flow): visit mass stays in their own community
    for n in nodes:
        if s_out[n] <= 0:
            P_within[idx[partition[n]]] += r[n]

    # Modularity Q = Σ (P_aa − p_a²)
    Q = float(np.sum(P_within - p ** 2))

    # Normalized cut Ncut = 1 − (1/K) Σ P_aa / p_a
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p > 0, P_within / p, 0.0)
    ncut = float(1.0 - ratio.sum() / K) if K > 0 else np.nan

    # map-equation codelength — evaluated under InfoMap's own self-consistent
    # flow model rather than the hand-rolled walk above (see docstring)
    L = _map_equation_codelength_infomap(
        G, partition, weight=weight, directed=G.is_directed(),
    )

    return {"modularity": Q, "ncut": ncut,
            "codelength": L, "n_communities": K}


def compare_partition_quality_hybrid(
    G: nx.DiGraph,
    partitions: dict,
    weight: str = "weight",
    alpha: float = 0.85,
) -> pd.DataFrame:
    """
    Score every partition with the three intrinsic course measures (Modularity,
    Ncut, map-equation codelength) plus its community count. PageRank is computed
    once on ``G`` and shared. NMI (the fourth measure) is pairwise: see
    :func:`compute_nmi_matrix`.

    Returns
    -------
    pd.DataFrame
        Rows = methods, columns = ``n_communities``, ``modularity``, ``ncut``,
        ``codelength``.
    """
    pr = nx.pagerank(G, alpha=alpha, weight=weight)
    rows = {}
    for name, part in partitions.items():
        rows[name] = score_partition_hybrid(
            G, part, pagerank=pr, weight=weight, alpha=alpha)
    df = pd.DataFrame(rows).T
    return df[["n_communities", "modularity", "ncut", "codelength"]]


def plot_partition_quality_hybrid(
    quality_df: pd.DataFrame,
    title: str = "",
    save: bool = True,
) -> None:
    """
    Point panels of the three intrinsic quality measures across methods, plus the
    community count to contextualise them. Ncut and the map-equation codelength
    are shown as ``1 − Ncut`` and ``− InfoMap`` so all three quality panels share
    the "higher better" orientation (same convention as the course slides).

    Each measure is drawn as a marker per method on an auto-scaled y-axis (same
    style as the course-slide comparison figure). Markers rather than bars-from-
    zero so the negative codelength (``− InfoMap``) reads as a ranking instead of
    a large downward bar, and so small differences in the bounded measures stay
    visible. Laid out 2×2 for slide use.
    """
    measures = [
        ("n_communities", "Communities found", lambda s: s),
        ("modularity",    "Modularity Q", lambda s: s),
        ("ncut",          "1 − Ncut", lambda s: 1.0 - s),
        ("codelength",    "− InfoMap", lambda s: -s),
    ]
    x = np.arange(len(quality_df))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, (col, lab, transform) in zip(axes.flat, measures):
        y = transform(quality_df[col].to_numpy())
        ax.plot(x, y, color="#5c6bc0", linewidth=1.5, alpha=0.55, zorder=2)
        ax.scatter(x, y, s=130, color="#5c6bc0", edgecolor="white",
                   linewidth=1.2, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(quality_df.index, rotation=30, ha="right")
        ax.set_title(lab)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.margins(y=0.18)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Community-detection quality: {title}" if title else "Community-detection quality")
    fig.tight_layout()
    if save:
        savefig(f"partition_quality_4measures_{_slug(title)}")
    plt.show()


# ====================================================================================
# Network layout: topology coloured by community (and topology-vs-geography)
# ====================================================================================


# ------------------------------------------------------------------------------------
# Interactive (Plotly) network layouts – zoom / pan / hover
# ------------------------------------------------------------------------------------


# ====================================================================================
# Mixed-Membership SBM via variational EM (Airoldi et al. 2008)
# ====================================================================================


def run_mmsbm_custom_hybrid(
    G: nx.Graph,
    K: int = 10,
    n_iter: int = 50,
    alpha: float | None = None,
    weight_mode: str = "poisson",
    tol: float = 1e-4,
    seed: int = 42,
    init: str = "louvain",
    verbose: bool = True,
) -> tuple[np.ndarray, Partition]:
    """
    Mixed-Membership Stochastic Blockmodel via variational EM
    (Airoldi, Blei, Fienberg & Xing 2008).

    Each node :math:`p` carries a Dirichlet-distributed membership vector
    :math:`\\pi_p \\in \\Delta^{K-1}`, and each ordered pair :math:`(p, q)`
    has latent block assignments :math:`z_{p \\to q}, z_{p \\leftarrow q}
    \\sim \\mathrm{Mult}(\\pi)`. The edge :math:`A_{pq}` is drawn from
    :math:`\\mathrm{Bernoulli}(B[z_{p \\to q}, z_{p \\leftarrow q}])`
    (``weight_mode='binary'``) or :math:`\\mathrm{Poisson}(B[z, z'])`
    on :math:`\\log(1 + w_{pq})`-transformed edge weights (``weight_mode='poisson'``,
    default). The log1p transform compresses the ~6-decade hybrid weight range
    into a numerically stable ~0–11 range while preserving relative ordering.

    Variational EM follows the original paper (eqs. 3, 5, 7) with full
    materialisation of the per-pair multinomials :math:`\\phi_{p \\to q}`
    and :math:`\\phi_{p \\leftarrow q}` as :math:`(N, N, K)` tensors –
    memory cost :math:`\\sim N^2 K` floats. For the hybrid 30 km Italy
    giant (:math:`N \\approx 1{,}800`) at :math:`K=10` this is :math:`\\sim`
    260 MB, runnable in a notebook. For larger networks, consider the
    stochastic-variational variant of Gopalan & Blei (2013).

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Network. Edge weights used iff ``weight_mode='poisson'``.
    K : int
        Number of blocks. For cross-validation with graph-tool's
        ``OverlapBlockState``, set this to graph-tool's auto-selected
        block count (read from its output CSV).
    n_iter : int
        Maximum variational EM iterations.
    alpha : float or None
        Dirichlet concentration hyperparameter. Default ``1/K`` encourages
        sparse memberships (most mass on one or two blocks).
    weight_mode : {'binary', 'poisson'}, default ``'poisson'``
        * ``'binary'`` – Bernoulli edge likelihood on the 0/1 adjacency.
          Loses the hybrid's continuous weight information but is the
          canonical Airoldi 2008 formulation.
        * ``'poisson'`` – Poisson likelihood on ``log1p(weight)``. Preserves
          relative weight ordering across the hybrid's ~6-decade range
          (min ≈ 0.02, max ≈ 4.8e4) by compressing to ~0–11, which keeps
          Poisson rates numerically stable. This matches the prof's algorithm
          slide (MM-SBM: weighted=YES) for the hybrid network.
    tol : float
        Convergence threshold (currently advisory only – fixed
        ``n_iter`` is used).
    seed : int
        RNG seed for initialisation.
    verbose : bool
        Print iteration summary (entropy of mean π, block matrix extrema).

    Returns
    -------
    pi : np.ndarray, shape (N, K)
        Expected membership probabilities under the variational posterior:
        :math:`\\hat\\pi_{p, k} = \\gamma_{p, k} / \\sum_k \\gamma_{p, k}`.
    hard_partition : dict
        ``{node_id: int}`` mapping (argmax of :math:`\\hat\\pi`) – suitable
        for NMI comparison against single-membership methods.

    References
    ----------
    Airoldi E.M., Blei D.M., Fienberg S.E. & Xing E.P. (2008). Mixed
    Membership Stochastic Blockmodels. *Journal of Machine Learning
    Research*, 9, 1981-2014.

    Gopalan P. & Blei D.M. (2013). Efficient discovery of overlapping
    communities in massive networks. *PNAS*, 110(36), 14534-14539.
    """
    from scipy.special import digamma

    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    N = len(nodes)
    node2idx = {n: i for i, n in enumerate(nodes)}

    if alpha is None:
        alpha = 1.0 / K
    if weight_mode not in ("binary", "poisson"):
        raise ValueError(f"weight_mode must be 'binary' or 'poisson', got {weight_mode!r}")

    # ── Adjacency matrix (directed; A[i,j] = edge i → j) ─────────────────────
    A = np.zeros((N, N), dtype=np.float32)
    for u, v, d in G.edges(data=True):
        i, j = node2idx[u], node2idx[v]
        if weight_mode == "binary":
            A[i, j] = 1.0
        else:
            A[i, j] = float(np.log1p(d.get("weight", 1.0)))

    if verbose:
        mem_mb = (N * N * K * 4 * 2) / 1024**2  # phi_send + phi_recv
        log.info("MMSB-EM: N=%d, K=%d, weight_mode=%s, est. memory %.0f MB",
                 N, K, weight_mode, mem_mb)
        log.info("  adjacency: %d edges, A.sum()=%.3g, A.mean()=%.4f",
                 int((A > 0).sum()), float(A.sum()), float(A.mean()))

    # ── Initial hard labels for symmetry breaking ────────────────────────────
    # Pure-random γ init traps the EM in a degenerate fixed point: when B is
    # uniform across blocks, the E-step cannot differentiate r values, so the
    # M-step keeps B uniform forever. We seed with a hard partition from
    # spectral / Louvain / random, then soften it.
    if init == "spectral":
        # K-means on top-K eigenvectors of the symmetrized adjacency
        from sklearn.cluster import KMeans
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        A_sym = (A + A.T) / 2.0
        # Add self-loops to ensure connectivity for the eigensolver
        np.fill_diagonal(A_sym, A_sym.sum(axis=1) / max(N, 1) + 1e-6)
        try:
            k_eig = min(K, N - 1)
            _, vecs = eigsh(csr_matrix(A_sym.astype(np.float64)), k=k_eig, which="LA")
            init_labels = KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(vecs)
        except Exception as e:
            log.warning("spectral init failed (%s), falling back to random", e)
            init_labels = rng.integers(0, K, size=N)
    elif init == "louvain":
        try:
            import leidenalg, igraph as ig
            G_und = G.to_undirected()
            G_und.remove_edges_from(nx.selfloop_edges(G_und))
            g_ig = ig.Graph(directed=False)
            g_ig.add_vertices(N)
            g_ig.add_edges([(node2idx[u], node2idx[v]) for u, v in G_und.edges()])
            part = leidenalg.find_partition(
                g_ig, leidenalg.RBConfigurationVertexPartition, seed=seed
            )
            init_labels = np.array(part.membership)
            # If Louvain found more/fewer than K, remap via K-means on indicator embedding
            if init_labels.max() + 1 != K:
                from sklearn.cluster import KMeans
                one_hot = np.eye(init_labels.max() + 1)[init_labels]
                init_labels = KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(one_hot)
        except Exception as e:
            log.warning("louvain init failed (%s), falling back to random", e)
            init_labels = rng.integers(0, K, size=N)
    else:  # "random"
        init_labels = rng.integers(0, K, size=N)

    # ── Variational parameters ───────────────────────────────────────────────
    # γ_p: high concentration on the initial label, small mass on others. Soft
    # enough that the EM can move nodes between blocks; hard enough that B is
    # immediately differentiated across blocks.
    gamma = np.full((N, K), alpha, dtype=np.float32)
    gamma[np.arange(N), init_labels] += 10.0
    gamma += rng.random((N, K)).astype(np.float32) * 0.1

    # φ[i, j, k] = ξ_{i→j, k} = per-pair multinomial – i's block when interacting
    # with j (the *sender* indicator for endpoint i). The receiver indicator for
    # the same pair is φ[j, i, k] = ξ_{j→i, k} – same tensor, transposed. There
    # is no separate "receive" tensor in Airoldi's formulation.
    # Initialise φ from the hard labels too (peaked at init_labels[i]).
    phi = np.full((N, N, K), 0.01, dtype=np.float32)
    for k in range(K):
        mask = (init_labels == k)
        phi[mask, :, k] = 0.9
    phi /= phi.sum(axis=2, keepdims=True)

    # Block interaction matrix – diagonal-dominant init so that block-pair
    # likelihoods differ from the start (constant-B init traps the EM in a
    # symmetric degenerate fixed point where all blocks are interchangeable).
    mean_density = float(A.mean()) if weight_mode == "binary" else max(float(A.mean()), 1e-3)
    B = np.full((K, K), mean_density * 0.5, dtype=np.float64)
    np.fill_diagonal(B, mean_density * 3.0)
    B *= 1.0 + rng.uniform(-0.1, 0.1, size=(K, K))
    if weight_mode == "binary":
        B = np.clip(B, 1e-3, 1.0 - 1e-3)
    else:
        B = np.maximum(B, 1e-3)

    # ── EM loop ──────────────────────────────────────────────────────────────
    for it in range(n_iter):
        Elog_pi = digamma(gamma) - digamma(gamma.sum(axis=1, keepdims=True))  # (N, K)

        # E-step: update φ[i, j, r] = ξ_{i→j, r}.
        # The receiver multinomial for the same pair is φ[j, i, s] = ξ_{j→i, s}.
        # `einsum('jis,rs->ijr', phi, logB)` computes Σ_s ξ_{j→i,s} · logB[r,s].
        if weight_mode == "binary":
            logB   = np.log(B + 1e-10)
            log1mB = np.log(1.0 - B + 1e-10)
            log_phi = (
                Elog_pi[:, None, :]
                + A[:, :, None]         * np.einsum('jis,rs->ijr', phi, logB)
                + (1.0 - A)[:, :, None] * np.einsum('jis,rs->ijr', phi, log1mB)
            )
        else:  # poisson
            logB = np.log(B + 1e-10)
            log_phi = (
                Elog_pi[:, None, :]
                + A[:, :, None] * np.einsum('jis,rs->ijr', phi, logB)
                - np.einsum('jis,rs->ijr', phi, B)
            )

        log_phi -= log_phi.max(axis=2, keepdims=True)  # numerical stability
        phi = np.exp(log_phi).astype(np.float32)
        phi /= phi.sum(axis=2, keepdims=True)

        # ── M-step ───────────────────────────────────────────────────────────
        # γ_{p,k} = α + Σ_q φ[p, q, k]   – Airoldi eq. 7
        gamma = (alpha + phi.sum(axis=1)).astype(np.float32)

        # B[r, s] = Σ_{ij} A[i,j] · φ[i,j,r] · φ[j,i,s] / Σ_{ij} φ[i,j,r] · φ[j,i,s]
        # – Airoldi eq. 6 (Bernoulli) / analogous for Poisson rate
        phi_T = phi.transpose(1, 0, 2)  # phi_T[i, j, s] = phi[j, i, s] = ξ_{j→i, s}
        numer = np.einsum('ijr,ijs->rs', phi * A[..., None], phi_T)
        denom = np.einsum('ijr,ijs->rs', phi, phi_T)
        B = (numer + 1e-10) / (denom + 1e-10)
        if weight_mode == "binary":
            B = np.clip(B, 1e-6, 1.0 - 1e-6)
        else:
            B = np.maximum(B, 1e-6)

        if verbose and (it % 5 == 0 or it == n_iter - 1):
            pi_tmp = gamma / gamma.sum(axis=1, keepdims=True)
            mean_entropy = float(
                -np.sum(pi_tmp * np.log(pi_tmp + 1e-10), axis=1).mean()
            )
            log.info(
                "  iter %3d/%d  mean π entropy=%.3f  B range=[%.3g, %.3g]",
                it + 1, n_iter, mean_entropy, float(B.min()), float(B.max()),
            )

    # ── Final: π = expected membership, hard partition = argmax ─────────────
    pi = gamma / gamma.sum(axis=1, keepdims=True)
    hard_partition = {nodes[i]: int(np.argmax(pi[i])) for i in range(N)}
    if verbose:
        log.info("MMSB-EM done: %d unique argmax blocks (of K=%d possible)",
                 len(set(hard_partition.values())), K)
    return pi, hard_partition


# ==============================================================================
# Temporal modularity evolution (sliding-window Q around a main shock)
# ==============================================================================
def build_window_network_hybrid(
    df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    cell_size_km: float,
    target_crs: str = "epsg:32632",
    alpha: float = 0.7,
    tau_days: float = 1.0,
    r0: float = 20.0,
    spatial_threshold_km: float = 300.0,
    time_threshold_sec: float = 72 * 3600,
) -> nx.DiGraph:
    """
    Build the hybrid Abe-Suzuki network for the events in a single time window.

    The catalog is sliced to ``start_time <= time < end_time`` and passed to
    :func:`src.network_custom.build_abe_suzuki_network_custom_hybrid` with the
    given decay/threshold parameters. Used by :func:`compute_q_over_time_hybrid`
    to rebuild the network for each sliding window.

    Parameters
    ----------
    df : pd.DataFrame
        Full catalog with a tz-aware ``time`` column (plus the columns the
        builder needs: ``latitude``, ``longitude``, ``depth_km``, ``magnitude``).
    start_time, end_time : pd.Timestamp
        Half-open window ``[start_time, end_time)``.
    cell_size_km : float
        Cubic cell side (km) for spatial discretisation.
    target_crs : str
        Projected CRS for distance computation (default ``epsg:32632``, Italy).
    alpha, tau_days, r0 : float
        Hybrid weighting parameters (magnitude exponent, temporal decay in days,
        spatial decay in km).
    spatial_threshold_km, time_threshold_sec : float
        Hard filters on the distance / elapsed time between consecutive events.

    Returns
    -------
    nx.DiGraph
        The windowed hybrid network (possibly empty if the slice is too small).
    """
    df_win = df[(df["time"] >= start_time) & (df["time"] < end_time)]
    return build_abe_suzuki_network_custom_hybrid(
        df_win,
        cell_size_km=cell_size_km,
        spatial_threshold_km=spatial_threshold_km,
        time_threshold_sec=time_threshold_sec,
        target_crs=target_crs,
        alpha=alpha,
        tau=tau_days * 86400.0,
        r0=r0,
        info=False,
    )


def compute_modularity_from_partition(
    G: nx.Graph,
    partition: Partition,
    weight: str | None = "weight",
    resolution: float = 1.0,
) -> float:
    """
    Newman modularity Q of a ``{node: community}`` partition.

    The graph is projected to a simple undirected graph (self-loops removed)
    before scoring, matching how :func:`run_louvain_hybrid` defines its
    partition. Nodes missing from ``partition`` are dropped.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Network whose modularity is measured.
    partition : dict
        ``{node: community_id}`` mapping (e.g. the output of
        :func:`run_louvain_hybrid`).
    weight : str or None
        Edge-data key for weighted modularity, or ``None`` for the unweighted
        count. Default ``"weight"``.
    resolution : float
        Resolution parameter γ of the (generalised) modularity. ``1.0`` is the
        classic Newman-Girvan definition.

    Returns
    -------
    float
        Modularity Q, or ``nan`` if the graph has no edges.
    """
    from collections import defaultdict

    G_und = G.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))
    if G_und.number_of_edges() == 0:
        return float("nan")

    groups: dict[int, set] = defaultdict(set)
    for node in G_und.nodes():
        if node in partition:
            groups[partition[node]].add(node)
    communities = list(groups.values())
    if not communities:
        return float("nan")

    return nx.algorithms.community.modularity(
        G_und, communities, weight=weight, resolution=resolution
    )


def compute_q_over_time_hybrid(
    df: pd.DataFrame,
    window_days: float,
    step_days: float,
    cell_size_km: float,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    alpha: float = 0.7,
    r0: float = 20.0,
    tau_days: float = 1.0,
    resolution: float = 1.0,
    target_crs: str = "epsg:32632",
    spatial_threshold_km: float = 300.0,
    time_threshold_sec: float = 72 * 3600,
    seed: int = 42,
    min_edges: int = 1,
) -> pd.DataFrame:
    """
    Sliding-window modularity Q(t) of the hybrid earthquake network.

    A window of width ``window_days`` is slid in ``step_days`` increments across
    ``[start_time, end_time)``. For each position the hybrid network is rebuilt
    from the events inside the window (:func:`build_window_network_hybrid`),
    partitioned with Louvain (:func:`run_louvain_hybrid`, Reichardt-Bornholdt at
    the given ``resolution``), and its modularity scored
    (:func:`compute_modularity_from_partition`). Following Abe & Suzuki, a sharp
    drop in Q marks the temporary collapse of community structure at a main shock.

    Each row is stamped with the **window centre** time, so a drop aligned with a
    main shock appears at ``t_relative ≈ 0`` when plotted against the event time.

    Parameters
    ----------
    df : pd.DataFrame
        Full catalog with a tz-aware ``time`` column.
    window_days : float
        Width of the sliding window (days).
    step_days : float
        Step between consecutive window starts (days).
    cell_size_km : float
        Cubic cell side (km).
    start_time, end_time : pd.Timestamp
        Range scanned; the last window is the one whose end does not exceed
        ``end_time``.
    alpha, r0, tau_days : float
        Hybrid weighting parameters passed to the per-window builder.
    resolution : float
        Louvain resolution γ (also used for the modularity score).
    target_crs : str
        Projected CRS (default ``epsg:32632``, Italy).
    spatial_threshold_km, time_threshold_sec : float
        Hard filters for the per-window builder.
    seed : int
        Louvain RNG seed.
    min_edges : int
        Windows with fewer than this many edges get ``Q = nan`` (too sparse to
        partition meaningfully).

    Returns
    -------
    pd.DataFrame
        Columns ``time`` (window centre, tz-aware), ``Q``, ``n_events``,
        ``n_nodes``, ``n_edges``, sorted by ``time``.
    """
    window = pd.Timedelta(days=window_days)
    step = pd.Timedelta(days=step_days)
    half = window / 2

    rows = []
    t0 = start_time
    while t0 + window <= end_time:
        t1 = t0 + window
        df_win = df[(df["time"] >= t0) & (df["time"] < t1)]
        n_events = len(df_win)

        G_win = build_window_network_hybrid(
            df, start_time=t0, end_time=t1, cell_size_km=cell_size_km,
            target_crs=target_crs, alpha=alpha, tau_days=tau_days, r0=r0,
            spatial_threshold_km=spatial_threshold_km,
            time_threshold_sec=time_threshold_sec,
        )
        n_nodes = G_win.number_of_nodes()
        n_edges = G_win.number_of_edges()

        if n_edges >= min_edges:
            part = run_louvain_hybrid(G_win, seed=seed, resolution=resolution)
            Q = compute_modularity_from_partition(
                G_win, part, weight="weight", resolution=resolution
            )
        else:
            Q = float("nan")

        rows.append({
            "time": t0 + half,
            "Q": Q,
            "n_events": n_events,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
        })
        t0 = t0 + step

    log.info("compute_q_over_time_hybrid: %d windows (width=%.0fd, step=%.0fd)",
             len(rows), window_days, step_days)
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)