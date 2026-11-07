# beast-compare

Record the Beast output of several ADS-B receivers side by side, then
compare them transmission by transmission: who decoded what, at what
range, who acquired an aircraft first and who held it longest.

It is built for receivers that share one antenna (through a splitter), so
every receiver sees the same transmissions and differences come from the
receiver chain: the SDR, its gain and attenuation, and its decoder.

## Commands

```sh
beast-compare record --config recorder.yaml
beast-compare compare /path/to/run
```

`record` runs until stopped, writing everything under one run directory.
`compare` analyses a recorded run offline.

## Recording

```yaml
run_id: 2026-11-baseline
data_dir: /opt/adsb/data/beast-compare
receivers:
  - name: airspy
    beast: 127.0.0.1:30105
    stats:
      file: /opt/adsb/data/airspy_adsb/stats.json
    metadata:
      sdr: Airspy R2
  - name: fa-proplus
    beast: 127.0.0.1:30106
    stats:
      url: http://127.0.0.1:8089/data/stats.json
    metadata:
      sdr: FlightAware Pro Stick Plus
      attenuation_db: 10
```

Each `beast` endpoint must be a single decoder's unreduced Beast output
(readsb's or airspy_adsb's port 30005), never an aggregator that merges
several receivers. `stats` is optional and is polled once a minute.

A run is defined by its configuration. Restarting with the same
configuration resumes the run; restarting with a different one is refused.
Change `run_id` to start a new run.

```text
<data_dir>/<run_id>/
    manifest.json             configuration and sessions
    events.jsonl              connects, disconnects, session start and stop
    <receiver>/
        YYYYMMDDTHH.bcr.zst   frames, hourly, zstd
        stats.jsonl           the decoder's stats
```

A `.bcr.zst` file is a sequence of zstd frames, one per flush (every
10 seconds by default). Decompressed, it is a run of records
`[u64 arrival_unix_ns][u8 len][type byte + unescaped Beast body]`,
big-endian. A crash loses at most the last flush interval.

## Development

```sh
nix develop          # or direnv allow
pytest
mypy src tests
nix flake check -L   # everything CI runs
```
