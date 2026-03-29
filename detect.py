# ============================================================
# Hardware Trojan Detector — Command Line Tool
# Usage: python detect.py --netlist path/to/folder
# ============================================================
import os, re, sys, argparse, torch, networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

# ── MODEL ────────────────────────────────────────────────────
class TrojanDetector(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.classifier = torch.nn.Linear(hidden, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.classifier(x)

# ── PARSER ───────────────────────────────────────────────────
def parse_to_graph(folder_path):
    G = nx.DiGraph()
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if not filename.endswith('.v'): continue
            if 'test' in filename.lower(): continue
            with open(os.path.join(root, filename), 'r', errors='ignore') as f:
                content = f.read()
            content = re.sub(r'//.*?\n', '\n', content)
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            kw = {'begin','end','if','else','case','posedge',
                  'negedge','always','assign','endmodule','module'}
            for match in re.finditer(r'\breg\b[^;]*?(\w+)\s*;', content):
                sig = match.group(1).strip()
                if sig and len(sig) > 1:
                    G.add_node(sig, node_type='reg')
            for match in re.finditer(r'assign\s+(\w+)\s*=\s*([^;]+);', content):
                lhs = match.group(1)
                G.add_node(lhs, node_type='wire')
                for sig in re.findall(r'\b([a-zA-Z_]\w*)\b', match.group(2)):
                    if sig and sig not in kw and not sig[0].isdigit():
                        G.add_edge(sig, lhs)
            for match in re.finditer(r'(\w+)\s*<=\s*([^;]+);', content):
                lhs = match.group(1)
                for sig in re.findall(r'\b([a-zA-Z_]\w*)\b', match.group(2)):
                    if sig and sig not in kw and not sig[0].isdigit():
                        if G.has_node(lhs) and G.has_node(sig):
                            G.add_edge(sig, lhs)
    isolated = [n for n in G.nodes() if G.degree(n)==0]
    G.remove_nodes_from(isolated)
    return G

# ── FEATURES ─────────────────────────────────────────────────
def extract_features(G):
    node_type_map = {'reg':0,'wire':1,'signal':2,'unknown':3}
    node_list = list(G.nodes())
    max_deg   = max((G.degree(n) for n in node_list), default=1)
    n_nodes   = G.number_of_nodes()
    n_edges   = G.number_of_edges()
    n_regs    = sum(1 for n in node_list if G.nodes[n].get('node_type')=='reg')
    n_wires   = sum(1 for n in node_list if G.nodes[n].get('node_type')=='wire')
    n_iso     = sum(1 for n in node_list if G.degree(n)==0)
    avg_deg   = sum(G.degree(n) for n in node_list)/n_nodes if n_nodes else 0
    density   = n_edges/(n_nodes*(n_nodes-1)) if n_nodes>1 else 0
    reg_ratio = n_regs/n_nodes if n_nodes else 0
    iso_ratio = n_iso/n_nodes if n_nodes else 0
    features  = []
    for node in node_list:
        ntype     = G.nodes[node].get('node_type','unknown')
        in_deg    = G.in_degree(node)
        out_deg   = G.out_degree(node)
        total_deg = in_deg + out_deg
        neighbors = list(G.predecessors(node)) + list(G.successors(node))
        avg_nb    = sum(G.degree(n) for n in neighbors)/len(neighbors) if neighbors else 0
        deg_anom  = abs(total_deg-avg_deg)/(avg_deg+1)
        features.append([
            node_type_map.get(ntype,3),
            in_deg, out_deg, total_deg,
            total_deg/max_deg,
            1 if ntype=='reg' else 0,
            1 if total_deg==0 else 0,
            1 if in_deg>out_deg else 0,
            1 if out_deg>in_deg else 0,
            avg_nb,
            n_nodes, n_edges, avg_deg,
            density, reg_ratio, iso_ratio,
            deg_anom, n_regs, n_wires
        ])
    return torch.tensor(features, dtype=torch.float)

def graph_to_pyg(G):
    node_list  = list(G.nodes())
    node_idx   = {n:i for i,n in enumerate(node_list)}
    x          = extract_features(G)
    edges      = [(node_idx[u],node_idx[v]) for u,v in G.edges()
                  if u in node_idx and v in node_idx]
    edge_index = torch.tensor(edges,dtype=torch.long).t().contiguous() \
                 if edges else torch.zeros((2,0),dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

# ── MAIN ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Hardware Trojan Detector — GNN-based Verilog netlist analyzer'
    )
    parser.add_argument('--netlist', required=True,
                        help='Path to folder containing Verilog (.v) files')
    parser.add_argument('--model', default='best_fold0.pt',
                        help='Path to trained model weights')
    args = parser.parse_args()

    # Check inputs
    if not os.path.exists(args.netlist):
        print(f"ERROR: Folder not found: {args.netlist}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)

    print("="*55)
    print("  HARDWARE TROJAN DETECTOR")
    print("  GNN-based Verilog Netlist Analyzer")
    print("="*55)
    print(f"\nAnalyzing: {args.netlist}")

    # Parse netlist
    print("Parsing Verilog files...")
    G = parse_to_graph(args.netlist)

    if G.number_of_nodes() == 0:
        print("ERROR: No parseable content found in Verilog files.")
        print("Make sure your .v files contain reg/assign/always statements.")
        sys.exit(1)

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load model
    print("Loading model...")
    model = TrojanDetector(in_channels=19, hidden=128, out_channels=2)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()

    # Run detection
    data = graph_to_pyg(G)
    with torch.no_grad():
        out   = model(data.x, data.edge_index,
                      torch.zeros(data.x.size(0), dtype=torch.long))
        probs = F.softmax(out, dim=1)
        pred  = out.argmax(dim=1).item()
        conf_clean  = probs[0][0].item() * 100
        conf_trojan = probs[0][1].item() * 100

    print("\n" + "="*55)
    if pred == 1:
        print("  🚨 RESULT: TROJAN DETECTED")
        print(f"  Confidence: {conf_trojan:.1f}%")
        print(f"  Clean probability:  {conf_clean:.1f}%")
        print(f"  Trojan probability: {conf_trojan:.1f}%")
    else:
        print("  ✅ RESULT: CLEAN — No Trojan Detected")
        print(f"  Confidence: {conf_clean:.1f}%")
        print(f"  Clean probability:  {conf_clean:.1f}%")
        print(f"  Trojan probability: {conf_trojan:.1f}%")
    print("="*55)
    print(f"\nNodes analyzed: {G.number_of_nodes()}")
    print(f"Edges analyzed: {G.number_of_edges()}")
    print("\nNote: This tool was trained on AES and RS232 Trust-Hub")
    print("benchmarks. Results on other chip types may vary.")

if __name__ == '__main__':
    main()
