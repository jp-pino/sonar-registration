# Mosaicing of Multibeam Sonar Feed Through Fourier-Based Registration

This repo accompanies the thesis by Juan Pablo Pino.

## Dependencies

A conda environment file is provided in the root directory. To create the environment, run:

```bash
conda env create -f environment.yml
```

## Enabling autocomplete

To enable autocomplete for the command line interface, run:

```bash
eval "$(register-python-argcomplete main_pipeline.py)"
```

## Running the pipeline

An example of how to run the pipeline:
```bash
./main_pipeline.py logs/multibeam_BYEDP210001_2024-07-25_132120.mbez --algorithm FOURIER_MELLIN --disable_loop_closure --disable_realignment --out out/fmt/harbor-entrance-final --output_frequency 10 --resize 0.1
```

Use the `--help` flag to see all available options.