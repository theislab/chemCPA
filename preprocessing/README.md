#### Clarifcation on the avaialable datasets

The `lincs_full.h5ad` as a combination of the available L1000 datasets from phase 1 and phase 2, available here: 
- L1000 Connectivity Map Phase I: [GSE70138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138)
- L1000 Connectivity Map Phase II: [GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)

We combined the data which is only available in the `.gctx` together with its metadata, cf. `<...>_inst_info.txt`, and make it available as scanpy compatible `.h5ad` object. 

Note that not all perturbations correspond to small molecules which is why we subsetted the data to only contain perturbation types `trt_cp` and `ctl_vehicle`, resulting in a total of 1034271 observations.

The provided data is normalised.