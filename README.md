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

## Development

```sh
nix develop          # or direnv allow
pytest
mypy src tests
nix flake check -L   # everything CI runs
```
