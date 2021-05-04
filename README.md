# MechRepoNet for Alzheimer's Disease

This repository starts the process of using MechRepoNet and it's various
endpoints to probe potential treatments for Alzheimer's Disease.

This repo uses the outputs of both [MechRepoNet](https://github.com/SuLab/MechRepoNet) and
[MRN new endpoints](https://github.com/SuLab/MRN_new_endpoints/).
This repo should be placed in the same directory as `MRN new endpoints` as it simlinks
to some of the param files found in that repository.

JSON files in the `0_data/manual` directory may need to be edited to provide the proper
paths for the data used from the MechRepoNet and pipeline

## Organization

This repository is organized as follows.

```
/0_data           # Contains data needed to for use within scripts
    manual        # Data built manually. Most will be included, unless built from proprietary source
    external      # Data acquired from external sources. Not included, but scripts will provide most
/1_code           # Contains all code for running the pipeline. Scripts and notebooks are numbered in order they should be run.
    tools         # contains tools for building
/2_pipeline       #  Output folder for pipeline. Not included with repo

```

## Setting up the environment

This repo uses essentially the same environment as [MechRepoNet](https://github.com/SuLab/MechRepoNet).
Please use the [requirements.txt](https://github.com/SuLab/MechRepoNet/blob/main/requirements.txt) from that
repository.

Additional requriement above MechRepoNet:
`$ pip install fpdf`

