#!/usr/bin/env python3
"""Generate Real Accra Road Network from OpenStreetMap (connected component only)"""
import os
import subprocess
import random
import xml.etree.ElementTree as ET
import networkx as nx

print("=" * 80)
print("GENERATING REAL ACCRA ROADS FROM OPENSTREETMAP")
print("=" * 80)

import osmnx as ox

# Focused area: Legon - Airport - Osu - Circle
north, south = 5.630, 5.545
east, west = -0.165, -0.225

print(f"\nDownloading Accra roads ({south}N-{north}N, {west}W-{east}W)...")

ox.settings.all_oneway = True
G_raw = ox.graph_from_bbox(bbox=(west, south, east, north), network_type='drive', simplify=False)
print(f"Raw download: {len(G_raw.nodes)} nodes, {len(G_raw.edges)} edges")

# Keep only the largest strongly connected component for routability
components = list(nx.strongly_connected_components(G_raw))
components.sort(key=len, reverse=True)
largest = components[0]
G = G_raw.subgraph(largest).copy()
print(f"Largest connected component: {len(G.nodes)} nodes, {len(G.edges)} edges")
print(f"  ({len(G.nodes)/len(G_raw.nodes)*100:.0f}% of original network)")

OUTPUT_DIR = './accra_osm_network'
os.makedirs(OUTPUT_DIR, exist_ok=True)

osm_file = f'{OUTPUT_DIR}/accra.osm'
ox.save_graph_xml(G, filepath=osm_file)
print(f"Saved OSM: {osm_file}")

print("\nConverting to SUMO network with netconvert...")
net_file = f'{OUTPUT_DIR}/accra.net.xml'

result = subprocess.run([
    'netconvert',
    '--osm-files', osm_file,
    '--output-file', net_file,
    '--geometry.remove',
    '--ramps.guess',
    '--junctions.join',
    '--tls.guess-signals',
    '--tls.discard-simple',
    '--output.street-names',
    '--output.original-names',
    '--default.speed', '13.89',
    '--junctions.corner-detail', '5',
    '--no-warnings',
], capture_output=True, text=True, timeout=300)

if result.returncode == 0:
    print(f"SUMO network created: {net_file}")
else:
    print(f"netconvert stderr (first 300 chars): {result.stderr[:300]}")

# Extract edges
print("\nExtracting road segments...")
edges = []
tree = ET.parse(net_file)
root = tree.getroot()
for edge in root.findall('.//edge'):
    eid = edge.get('id')
    if eid and not eid.startswith(':') and edge.find('lane') is not None:
        edges.append(eid)
print(f"Found {len(edges)} usable road segments")

print(f"\nDONE stage 1. Files in {OUTPUT_DIR}/")
print(f"  Connected nodes: {len(G.nodes)}, Connected edges: {len(G.edges)}, SUMO edges: {len(edges)}")
