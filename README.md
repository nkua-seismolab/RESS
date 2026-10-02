# RESS

[![lint](https://github.com/nkua-seismolab/RESS/actions/workflows/lint.yml/badge.svg)](https://github.com/nkua-seismolab/RESS/actions/workflows/lint.yml)
[![tests](https://github.com/nkua-seismolab/RESS/actions/workflows/tests.yml/badge.svg)](https://github.com/nkua-seismolab/RESS/actions/workflows/tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

Near real-time shear-wave splitting analysis for SeisComP.

RESS is part of a four-application workflow for automatic weak-event source characterization:

[REPOL](https://github.com/nkua-seismolab/REPOL) / **[RESS](https://github.com/nkua-seismolab/RESS)** -> [REHASH](https://github.com/nkua-seismolab/REHASH) / [REBayFM](https://github.com/nkua-seismolab/REBayFM)

## How it works

RESS connects to the SeisComP messaging system and listens for new events. For
every S pick it:

1. Performs pre-processing quality control (e.g., shear-wave window criterion).
1. Fetches three-component waveforms from a local SDS archive.
2. Runs splitting analysis with the eigenvalue [(Silver & Chan, 1991)](https://doi.org/10.1029/91JB00899) 
   and the rotation-correlation [(Bowman & Ando, 1987)](https://doi.org/10.1111/j.1365-246X.1987.tb01367.x)
   methods, over a bank of bandpass filters (adapted from [Savage et al., 2010](https://doi.org/10.1029/2010JB007722))
   and a set of analysis windows around the S pick, with cluster analysis selecting 
   the best. [(Teanby et al., 2004)](https://doi.org/10.1785/0120030123).
3. Grades each measurement with a quality index
   [(Wuestefeld et al., 2010)](https://doi.org/10.1111/j.1365-2478.2010.00891.x) into
   Good Split, Good Null and Poor / Noise.

Results - fast S-wave polarization direction, time-delay, unsplit S-wave polarization, and
quality - are written as a JSON comment on the S pick
(`<pickID>/comment/ress#<time>`) and stored in a PostgreSQL database. A
bundled FastAPI service exposes the database for retrieval, to permit acquisition of detailed
splitting information (which would bloat the SeisComP database).
[REBayFM](https://github.com/nkua-seismolab/REBayFM) consumes the S-wave
polarizations downstream.

## Requirements

- A running SeisComP system and a local SDS waveform archive.
- Docker Engine with the Compose plugin.
- Resources: ~2 GB disk for the images (core & API share one image, plus
  PostgreSQL); roughly 2 GB RAM during processing.

Tested with SeisComP 7.3.0 on Ubuntu 24.04.1.

## Installation

```bash
git clone https://github.com/nkua-seismolab/RESS.git
cd RESS
cp config.example.yaml config.yaml
cp .env.template .env
```

Then:

1. Edit `config.yaml`: set `seiscomp.host` and `seiscomp.database` to your
   SeisComP messaging host and database URL.
2. Edit `.env`: set the PostgreSQL credentials (`POSTGRES_PASSWORD` at
   minimum). The database is created automatically on first start.
3. Edit `docker-compose.yml`: point the SDS archive volume at your archive
   (the container path `/data/archive` must match `sds.archive` in
   `config.yaml`).
4. Start the stack (worker, API, PostgreSQL):

   ```bash
   docker compose up -d --build
   docker compose logs -f ress
   ```

## Configuration

All options live in `config.yaml` and are documented inline in
[config.example.yaml](config.example.yaml). The main sections:

| Section | Purpose |
| ------- | ------- |
| `seiscomp` | Messaging host, database URL, wait time, reprocessing policy |
| `sds` | Path of the SDS waveform archive inside the container |
| `velocity_model` | 1-D velocity model used for ray tracing |
| `waveform` | Acquisition window, sampling rate, gap handling |
| `sws` | Shear-wave window, filter bank, analysis windows, clustering, quality |
| `processing` | Number of cores for the window analysis |
| `logging` | Level and optional log file |

PostgreSQL connection settings stored in `.env`.

## Output

- A JSON comment per measured S pick (id `<pickID>/comment/ress#<time>`)
  with the splitting parameters, quality, and provenance.
- A PostgreSQL record per measurement, served by the API:

  ```bash
  curl http://localhost:8070/health
  curl "http://localhost:8070/results?quality_class=Good%20Split"  # CSV
  ```

## Getting started

See [GETTING_STARTED.md](GETTING_STARTED.md) for a step-by-step demo run with
the example event `nkua2020abcd` from the demo dataset
([doi:10.5281/zenodo.23105466](https://doi.org/10.5281/zenodo.23105466)).

## License

[GPL-3.0](LICENSE)

## Funding

This work is part of the [TRANSFORM²](https://www.transform2-project.eu/) project which aims to improve physical and digital infrastructure across Near-Fault Observatories (NFOs) in Europe.

TRANSFORM² is funded by the European Union under project number 101188365 within the HORIZON-INFRA-2024-DEV-01-01 call.

<div align="center">
  <img src="https://www.transform2-project.eu/wp-content/uploads/2022/08/Logo_TRANSFORM2-round-logo-100x100-1.png" alt="TRANSFORM² logo">
</div>
