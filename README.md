# Overview 

This repository contains the modified recombination emulator and some accompanying code used in the paper ["Reconstructing the recombination history by combining early and late cosmological probes"](https://arxiv.org/abs/2404.05715) 

The trained tensorflow model is contained in the folder `trained_models/modrec_extended_34`, and the file `TrainedEmulator.py` has a convenience wrapper. This emulator was trained using an older version of the [CONNECT](https://github.com/AarhusCosmology/connect_public) framework.

# Usage

The notebook `example.ipynb` has a brief example of how to load the emulator and get predictions using the wrapper. This wrapper is intended for use outside of MCMC sampler. To use in an MCMC, use the MCMC plugin, originally provided from CONNECT.

Due to some changes between CONNECT versions with how models are saved, there may be some backwards compatibility issues if you try to use this emulator with the MCMC plugin provided by the latest CONNECT version. As such, this repository also contains an older version of the CONNECT plugin for [COBAYA](https://cobaya.readthedocs.io/en/latest/), which should work for this emulator. To ensure this works, you should modify `mcmc_plugin/connect.conf` to point towards the correct paths. In your COBAYA input file, your theory block should look like

```yaml
theory:
  CosmoConnect:
    extra_args:
      connect_model: modrec_extended_34
    ignore_obsolete: true
    path: /path/to/mcmc_plugin/cobaya
    python_path: /path/to/mcmc_plugin/cobaya/
```

# Citation
If you use this emulator, please consider citing the original ModRec paper:

```
@article{Lynch:2024gmp,
    author = "Lynch, Gabriel P. and Knox, Lloyd and Chluba, Jens",
    title = "{Reconstructing the recombination history by combining early and late cosmological probes}",
    eprint = "2404.05715",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "4",
    year = "2024"
}
```

Also consider citing the CONNECT paper, especially if you use the MCMC plugin:

```
@article{Nygaard:2022wri,
    author = "Nygaard, Andreas and Holm, Emil Brinch and Hannestad, Steen and Tram, Thomas",
    title = "{CONNECT: a neural network based framework for emulating cosmological observables and cosmological parameter inference}",
    eprint = "2205.15726",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1088/1475-7516/2023/05/025",
    journal = "JCAP",
    volume = "05",
    pages = "025",
    year = "2023"
}
```
