Upstream ST-TR (Spatial Temporal Transformer for skeleton action recognition)
================================================================================

Source: https://github.com/Chiaraplizz/ST-TR (MIT License)

Vendored at backend/third_party/ST-TR for IsoCourt training via
core/st_tr_official.py.

IsoCourt-specific additions:
- code/st_gcn/graph/mediapipe_blazepose.py — 33-node BlazePose graph (spatial partition A).
- Small device-agnostic patches in unit_gcn.py, unit_agcn.py, gcn_attention.py,
  spatial_transformer.py, temporal_transformer.py (replace legacy .cuda() calls).
