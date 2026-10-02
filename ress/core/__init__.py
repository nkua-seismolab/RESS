"""SeisComP-free scientific pipeline for shear-wave splitting analysis.

Implements the RESS core logic: waveform QC, ray-path screening,
adaptive filtering, per-window EV/RC splitting measurements, cluster
analysis and quality assessment. Only the boundary dataclasses in
:mod:`ress.core.models` are exchanged with the scclient layer.
"""
