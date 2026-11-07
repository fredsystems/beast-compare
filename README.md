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

## Comparing

```sh
beast-compare compare /path/to/run --lat 35.0 --lon -106.6 --alt-m 1600 \
    [--start 2026-11-08T00:00] [--end 2026-11-10T00:00] [--reference airspy]
```

The antenna position is given here rather than recorded, so that it never
ends up in a run directory or a configuration file. The first configured
receiver is the reference clock unless `--reference` names another.

The report directory (by default `<run>/reports/<start>_<end>/`) holds
`report.html`, a single self-contained page, and every table behind it as
parquet, including `transmissions.parquet`: one row per transmission, with
who decoded it, at what signal, and where the aircraft was.

How it works:

1. Every receiver's counter is mapped onto the reference receiver's from
   long messages both decoded once (robustly: identical content transmitted
   twice pairs up wrongly and is rejected). Aligned timestamps agree to a
   fraction of a microsecond.
2. Frames with the same bytes within 2 us on that clock are one
   transmission. The union of all transmissions is the reference set.
3. Airborne positions are decoded from CPR pairs, and every transmission of
   a known aircraft is placed on its track, so a receiver is credited at the
   range it heard an aircraft from even for short replies.
4. Each receiver is scored only over the minutes it was valid: connected,
   aligned, not dropping samples, and not overloaded (judged from readsb's
   noise floor and strong-message share, not from its gain).
5. Every rate carries a 95 % interval from resampling aircraft visits.

Long runs are processed an hour at a time (`--window-minutes`): frames,
the bulk of the data, are only ever held for one window. What is kept for
the whole run is per transmission, about 70 bytes each, so a 48-hour run
of four receivers needs a few GB.

## Development

```sh
nix develop          # or direnv allow
pytest
mypy src tests
nix flake check -L   # everything CI runs
```
