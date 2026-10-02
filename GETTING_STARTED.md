# Getting started with RESS

This walkthrough runs RESS on one demo earthquake using the demo dataset
([doi:10.5281/zenodo.23105466](https://doi.org/10.5281/zenodo.23105466)).

RESS is the second component of the weak-event source characterization workflow
(REPOL / RESS -> REHASH / REBayFM). It runs fine on its own; start the other
containers too if you want the full workflow from the same dispatch.

## Prerequisites

- A running SeisComP system with messaging and database.
- Docker Engine with the Compose plugin.
- The demo dataset files.
- The velocity model shipping with RESS.

## 1. Prepare SeisComP

Import the demo inventory into your SeisComP test system.

## 2. Prepare RESS

```bash
git clone https://github.com/nkua-seismolab/RESS.git
cd RESS
cp config.example.yaml config.yaml
cp .env.template .env
```

- In `config.yaml`, set `seiscomp.host` and `seiscomp.database` (defaults may
  suit a default SeisComP installation).
- In `.env`, set the PostgreSQL credentials (`POSTGRES_PASSWORD` at
  minimum).
- Unzip `archive.zip` and point the archive volume in `docker-compose.yml`
  at it, e.g.:

  ```yaml
      - /path/to/unzipped/archive:/data/archive:ro
  ```

Start the stack (core/worker, API, PostgreSQL) and watch the worker logs:

```bash
docker compose up -d --build
docker compose logs -f ress
```

Wait until you see that RESS connected and is watching for events.

## 3. Dispatch the demo event

On the SeisComP host:

```bash
scdispatch -i nkua2020abcd_crl_scml.xml -v
```

RESS picks up the new event, waits `seiscomp.wait_time`, selects the S picks
inside the shear-wave window, and runs the splitting analysis. Progress
appears in the container logs.

## 4. Check the results

- Each analyzed S pick receives a JSON comment
  (id `<pickID>/comment/ress#<time>`) with the splitting parameters and
  quality class.
- Query the API for the stored measurements:

  ```bash
  curl http://localhost:8070/health
  curl "http://localhost:8070/results"                              # CSV, all
  curl "http://localhost:8070/results?quality_class=Good%20Split"   # filtered
  ```

## 5. Rerun

To remove the dispatched objects and run the demo again:

```bash
scdispatch -i nkua2020abcd_crl_scml.xml -v -O remove
```

Set `seiscomp.reprocess: true` in `config.yaml` (and restart the container)
if you want RESS to reprocess events it has already handled.
