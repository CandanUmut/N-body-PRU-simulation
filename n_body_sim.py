#!/usr/bin/env python3
# =============================================================
#  🌌  PRU – 3‑D Relational Universe • Live Observatory Edition
# =============================================================
import numpy as np, networkx as nx, matplotlib.pyplot as plt, math, time
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ─────────────────────────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────────────────────────
INIT_NODES          = 1000
NODE_LIMIT          = 1000
REL_RADIUS          = 180.0

CREATION_TH         = 1.618
CONFLICT_TH         = 0.38
ISOLATION_GRACE     = 30

FREQ_SCALE          = 2.0
FORCE_SCALE         = 0.08
MAX_STEP_DIST       = 6.0
FREE_WILL_STD       = 0.025
TIMESTEPS           = 4_000

EDGE_ATTR_COLOR = "#18e3c9"
EDGE_REP_COLOR  = "#d94cd9"
CMAP            = plt.cm.viridis

np.set_printoptions(precision=3, suppress=True)   # nicer console dumps

# ─────────────────────────────────────────────────────────────
# 🛰️  KD‑tree neighbour search
# ─────────────────────────────────────────────────────────────
def build_finder():
    try:
        from scipy.spatial import cKDTree
        def neigh(node_id, pos_dict, r):
            keys = list(pos_dict)
            pts  = np.array([pos_dict[k] for k in keys])
            tree = cKDTree(pts)
            idxs = tree.query_ball_point(pos_dict[node_id], r)
            idxs.remove(keys.index(node_id))
            return [keys[i] for i in idxs]
        return neigh
    except ImportError:
        def brute(nid, posd, r):
            p0 = posd[nid]
            return [k for k, p in posd.items() if k != nid and np.linalg.norm(p - p0) < r]
        return brute
get_neigh = build_finder()

# ─────────────────────────────────────────────────────────────
# 🌱  INITIALISE UNIVERSE
# ─────────────────────────────────────────────────────────────
G = nx.Graph()
for idx, pos in enumerate(np.random.uniform(-450, 450, (INIT_NODES, 3))):
    G.add_node(idx,
        pos=pos,
        resonance=np.random.uniform(0.5, 1.0),
        truth=np.random.uniform(0.3, 0.9),
        polarity=np.random.choice([-1, 1]),
        alignment=np.random.uniform(-1, 1),
        energy=np.random.uniform(0.3, 1.0),
        frequency=np.random.uniform(0.5, 10.0),
        isolation=0,
    )
def connect_world():
    pos = {n:d["pos"] for n,d in G.nodes(data=True)}
    for n in list(G):
        for m in get_neigh(n, pos, REL_RADIUS):
            G.add_edge(n,m)
connect_world()

# ─────────────────────────────────────────────────────────────
# 🔄  CORE PHYSICS
# ─────────────────────────────────────────────────────────────
def step():
    pos = {n: d["pos"] for n, d in G.nodes(data=True)}
    new_nodes, kill = [], []

    for n, d in list(G.nodes(data=True)):
        neigh = list(G.neighbors(n))
        d["isolation"] = 0 if neigh else d["isolation"] + 1

        move = np.zeros(3)
        res_sum = tru_sum = weight_sum = 0.0

        for m in neigh:
            md = G.nodes[m]
            vec = md["pos"] - d["pos"];  dist = np.linalg.norm(vec)+1e-6
            dirn = vec / dist

            # signed relational weight for **motion**
            fh   = math.exp(-abs(d["frequency"] - md["frequency"]) / FREQ_SCALE)
            xi   = ((d["energy"] * md["energy"] * fh *
                    (1 + d["polarity"] * md["polarity"] * md["alignment"]))
                    - (1 - fh)) * FORCE_SCALE
            move += xi * dirn
            G[n][m]["xi"] = xi

            # ---------- positive weight for learning ----------
            w = abs(xi) + 1e-6          # always positive
            res_sum   += md["resonance"] * w
            tru_sum   += md["truth"]     * w
            weight_sum += w

        # cap displacement + jitter
        L = np.linalg.norm(move)
        if L > MAX_STEP_DIST:
            move = move / L * MAX_STEP_DIST
        d["pos"] += move + np.random.normal(0, FREE_WILL_STD, 3)

        # gentle frequency drift (entropy engine)
        d["frequency"] = np.clip(d["frequency"] +
                                 np.random.normal(0, 0.01), 0.3, 10.0)

        # attribute blending
        if weight_sum:
            target_res = res_sum / weight_sum
            target_tru = tru_sum / weight_sum
            d["resonance"] += 0.05 * (target_res - d["resonance"])
            d["truth"]     += 0.05 * (target_tru - d["truth"])

        # floor values so H doesn’t collapse
        d["resonance"] = np.clip(d["resonance"], 0.2, 2)
        d["truth"]     = np.clip(d["truth"],     0.2, 2)

        # birth / death
        H = d["resonance"] * d["truth"]
        if H > CREATION_TH and G.number_of_nodes() < NODE_LIMIT:
            nid = max(G) + 1
            G.add_node(
                nid,
                pos=d["pos"] + np.random.normal(0, 8, 3),
                resonance=d["resonance"] * 0.97,
                truth=d["truth"] * 0.97,
                polarity=-d["polarity"],
                alignment=-d["alignment"],
                energy=max(0.25, np.random.normal(d["energy"], 0.05)),
                frequency=max(0.3, np.random.normal(d["frequency"], 0.3)),
                isolation=0,
            )
            new_nodes.append(nid)
        elif d["isolation"] > ISOLATION_GRACE and H < CONFLICT_TH:
            kill.append(n)

    if kill:
        G.remove_nodes_from(kill)
    if new_nodes:
        for nid in new_nodes:
            connect_new(nid)
    rewire()

def connect_new(nid):
    pos={n:d["pos"] for n,d in G.nodes(data=True)}
    for m in get_neigh(nid,pos,REL_RADIUS): G.add_edge(nid,m)

def rewire():
    pos={n:d["pos"] for n,d in G.nodes(data=True)}
    G.remove_edges_from([(u,v) for u,v in G.edges() if np.linalg.norm(pos[u]-pos[v])>REL_RADIUS])
    for n in list(G):
        for m in get_neigh(n,pos,REL_RADIUS):
            if not G.has_edge(n,m): G.add_edge(n,m)

# ─────────────────────────────────────────────────────────────
# 🖼️  VISUAL & METRIC PANEL
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10,7), facecolor="black")
ax  = fig.add_subplot(111, projection="3d", facecolor="black")
ax.set_axis_off()
# static star‑field (collection 0)
star_xyz=np.random.uniform(-1_000,1_000,(800,3))
ax.scatter(star_xyz[:,0],star_xyz[:,1],star_xyz[:,2],
           s=1,c="white",alpha=.25,depthshade=False)

panel = ax.text2D(0.02,0.96,"",transform=ax.transAxes,
                  color="white",fontsize=8,family="monospace",va="top")

def metrics():
    N,E = G.number_of_nodes(), G.number_of_edges()
    cc_sizes = [len(c) for c in nx.connected_components(G)]
    largest = max(cc_sizes) if cc_sizes else 0
    mean_deg = 2*E/N if N else 0
    harmonies = [d["resonance"]*d["truth"] for _,d in G.nodes(data=True)]
    avg_H = np.mean(harmonies) if harmonies else 0
    freq = np.array([d["frequency"] for _,d in G.nodes(data=True)])
    entropy = (-(freq/ freq.sum() + 1e-9)*np.log(freq/ freq.sum()+1e-9)).sum()
    attr_edges = sum(1 for _,_,ed in G.edges(data=True) if ed.get("xi",0)>=0)
    repel_ratio = 1-attr_edges/E if E else 0
    return dict(N=N,E=E,lcc=largest,deg=mean_deg,
                H=avg_H,repel=repel_ratio,entropy=entropy)

def draw(frame):
    t0=time.time()
    step()
    # wipe dynamic artists
    while len(ax.collections)>1: ax.collections[-1].remove()

    # nodes
    pos_arr=np.array([d["pos"] for _,d in G.nodes(data=True)])
    freqs=np.array([d["frequency"] for _,d in G.nodes(data=True)])
    span=np.ptp(freqs) or 1.0
    colors=CMAP((freqs-freqs.min())/(span+1e-6))
    sizes=50*np.array([d["truth"] for _,d in G.nodes(data=True)])
    ax.scatter(pos_arr[:,0],pos_arr[:,1],pos_arr[:,2],
               s=sizes,c=colors,depthshade=True,edgecolors="none")

    # edges
    seg_attr,seg_rep=[],[]
    for u,v,ed in G.edges(data=True):
        (seg_attr if ed.get("xi",0)>=0 else seg_rep).append([G.nodes[u]["pos"],G.nodes[v]["pos"]])
    if seg_attr: ax.add_collection3d(Line3DCollection(seg_attr,colors=EDGE_ATTR_COLOR,lw=.8,alpha=.6))
    if seg_rep:  ax.add_collection3d(Line3DCollection(seg_rep, colors=EDGE_REP_COLOR, lw=.8,alpha=.6))

    # camera & limits
    ax.view_init(elev=25,azim=frame*.4%360)
    for lim in (-600,600),: ax.set_xlim3d(*lim); ax.set_ylim3d(*lim); ax.set_zlim3d(*lim)

    # metrics panel
    m=metrics()
    txt = (f"⏱ frame {frame:5d}\n"
           f"🪐 nodes  : {m['N']:4d}\n"
           f"🔗 edges  : {m['E']:4d}\n"
           f"🌳 LCC    : {m['lcc']:4d}\n"
           f"⌁ ⟨deg⟩   : {m['deg']:5.2f}\n"
           f"⚖ avg H  : {m['H']:5.2f}\n"
           f"🚫 repulse: {100*m['repel']:5.1f}%\n"
           f"🎶 entropy: {m['entropy']:5.2f}")
    panel.set_text(txt)

    # title shows real‑time FPS
    fps = 1/(time.time()-t0)
    ax.set_title(f"PRU Observatory  |  {fps:4.1f} FPS",color="white",fontsize=9)

ani = FuncAnimation(fig, draw, frames=TIMESTEPS, interval=45, blit=False)
plt.show()
