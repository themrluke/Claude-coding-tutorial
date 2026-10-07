"""A toy silicon tracker: charged particles curling in a solenoid, leaving hits on barrel layers.

This module is *provided*: you do not need to change it. It stands in for TrackML
so every exercise runs on a laptop in seconds. The output looks like TrackML after
hepattn's ``prep.py``: one hits table and one particles table per event, lengths in
millimetres, momenta in GeV, ``particle_id == 0`` for noise hits.

The physics, in one paragraph: a particle with transverse momentum pT (GeV) and
charge q in a field B (T) moves on a circle of radius ``R = pT / (0.3 B)`` metres
in the x-y plane. It crosses a cylinder of radius r at azimuth
``phi(r) = phi0 - q * asin(r / 2R)`` after a transverse path length
``s = 2R asin(r / 2R)``, and along the beam it moves ``z = vz + s * sinh(eta)``.
Particles with ``2R < r`` curl up before reaching that layer. We add measurement
smearing, a per-hit detection efficiency and random noise hits.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Geometry:
    """Barrel layers, loosely modelled on the TrackML pixel (volume 8) and short-strip (volume 13) barrels."""

    layer_radii_mm: tuple[float, ...] = (32.0, 72.0, 116.0, 172.0, 260.0, 360.0, 500.0, 660.0)
    layer_half_length_mm: tuple[float, ...] = (500.0, 500.0, 500.0, 500.0, 1100.0, 1100.0, 1100.0, 1100.0)
    volume_ids: tuple[int, ...] = (8, 8, 8, 8, 13, 13, 13, 13)
    rphi_resolution_mm: tuple[float, ...] = (0.015, 0.015, 0.015, 0.015, 0.05, 0.05, 0.05, 0.05)
    z_resolution_mm: tuple[float, ...] = (0.05, 0.05, 0.05, 0.05, 1.0, 1.0, 1.0, 1.0)
    b_field_tesla: float = 2.0
    hit_efficiency: float = 0.98

    @property
    def num_layers(self) -> int:
        return len(self.layer_radii_mm)


@dataclass(frozen=True)
class EventConfig:
    """What each event contains."""

    mean_num_particles: float = 25.0
    pt_min_gev: float = 0.4
    pt_scale_gev: float = 1.5  # pT = pt_min + Exponential(pt_scale)
    max_abs_eta: float = 3.0
    vertex_z_sigma_mm: float = 50.0
    noise_fraction: float = 0.1  # number of noise hits = Poisson(noise_fraction * number of signal hits)


def event_name(event_id: int) -> str:
    """TrackML-style event name, e.g. ``event000000042``."""
    return f"event{event_id:09d}"


def simulate_event(
    event_id: int,
    geometry: Geometry | None = None,
    config: EventConfig | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate one event. The same ``(seed, event_id)`` always gives the same event.

    Returns:
        hits: one row per hit, columns hit_id, x, y, z (mm), volume_id, layer_id, particle_id.
            Rows are shuffled, as in real data: nothing may rely on hit order.
        particles: one row per particle, columns particle_id, vx, vy, vz (mm), px, py, pz (GeV), q, nhits.
    """
    geometry = geometry or Geometry()
    config = config or EventConfig()
    rng = np.random.default_rng([seed, event_id])

    n = max(1, int(rng.poisson(config.mean_num_particles)))
    pt = config.pt_min_gev + rng.exponential(config.pt_scale_gev, n)
    eta = rng.uniform(-config.max_abs_eta, config.max_abs_eta, n)
    phi0 = rng.uniform(-np.pi, np.pi, n)
    q = rng.choice(np.array([-1, 1]), n)
    vz = rng.normal(0.0, config.vertex_z_sigma_mm, n)
    particle_id = event_id * 1_000_000 + np.arange(1, n + 1)

    radius_mm = 1000.0 * pt / (0.3 * geometry.b_field_tesla)  # helix radius

    hit_tables = []
    for layer, r in enumerate(geometry.layer_radii_mm):
        reaches = r < 2 * radius_mm
        arg = np.where(reaches, r / (2 * radius_mm), 0.0)
        bend = np.arcsin(arg)
        s_transverse = 2 * radius_mm * bend
        z = vz + s_transverse * np.sinh(eta)
        phi = phi0 - q * bend
        detected = reaches & (np.abs(z) < geometry.layer_half_length_mm[layer]) & (rng.random(n) < geometry.hit_efficiency)
        k = int(detected.sum())
        phi_meas = phi[detected] + rng.normal(0.0, geometry.rphi_resolution_mm[layer] / r, k)
        z_meas = z[detected] + rng.normal(0.0, geometry.z_resolution_mm[layer], k)
        hit_tables.append(
            pd.DataFrame(
                {
                    "x": r * np.cos(phi_meas),
                    "y": r * np.sin(phi_meas),
                    "z": z_meas,
                    "volume_id": geometry.volume_ids[layer],
                    "layer_id": layer,
                    "particle_id": particle_id[detected],
                }
            )
        )

    signal = pd.concat(hit_tables, ignore_index=True)

    n_noise = int(rng.poisson(config.noise_fraction * len(signal)))
    noise_layer = rng.integers(0, geometry.num_layers, n_noise)
    noise_r = np.asarray(geometry.layer_radii_mm)[noise_layer]
    noise_phi = rng.uniform(-np.pi, np.pi, n_noise)
    noise_hl = np.asarray(geometry.layer_half_length_mm)[noise_layer]
    noise = pd.DataFrame(
        {
            "x": noise_r * np.cos(noise_phi),
            "y": noise_r * np.sin(noise_phi),
            "z": rng.uniform(-noise_hl, noise_hl),
            "volume_id": np.asarray(geometry.volume_ids)[noise_layer],
            "layer_id": noise_layer,
            "particle_id": np.zeros(n_noise, dtype=np.int64),
        }
    )

    hits = pd.concat([signal, noise], ignore_index=True)
    hits = hits.iloc[rng.permutation(len(hits))].reset_index(drop=True)
    hits.insert(0, "hit_id", np.arange(1, len(hits) + 1))
    hits = hits.astype({"x": np.float32, "y": np.float32, "z": np.float32, "volume_id": np.int32, "layer_id": np.int32, "particle_id": np.int64})

    nhits = signal["particle_id"].value_counts()
    particles = pd.DataFrame(
        {
            "particle_id": particle_id.astype(np.int64),
            "vx": np.zeros(n, dtype=np.float32),
            "vy": np.zeros(n, dtype=np.float32),
            "vz": vz.astype(np.float32),
            "px": (pt * np.cos(phi0)).astype(np.float32),
            "py": (pt * np.sin(phi0)).astype(np.float32),
            "pz": (pt * np.sinh(eta)).astype(np.float32),
            "q": q.astype(np.int32),
        }
    )
    particles["nhits"] = particles["particle_id"].map(nhits).fillna(0).astype(np.int32)
    return hits, particles
