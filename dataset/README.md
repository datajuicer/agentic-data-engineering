# Dataset Utilities

This directory is organized by datasource. Put each datasource's conversion,
sampling, validation, and inspection scripts in its own subdirectory:

```text
dataset/
  openthoughts/
  openscience_reasoning_2/
```

Raw downloaded files should stay under `data/raw_data/<datasource>/`. Derived
training pools or processed outputs should stay under `data/<datasource>/` or a
clearly named output directory.
