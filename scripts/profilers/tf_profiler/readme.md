# TensorFlow Model Profiler

## Requirements
`tensorflow1.3`

## Set up
Make sure you add CUDA libcupti to your path:

`export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH`

---
From the top level mlod folder:
```
python scripts/profilers/tf_profiler/model_profiler.py
```

This will output number of parameters, shapes and memory information on the console.

---
## Viewing Data
To view the timeline information:
```
Open a Chrome Browser, type URL chrome://tracing, and load the json file.
```

Example Output:
```
Add                          1488.25MB (100.00%, 3.74%),        9.34ms (100.00%, 0.25%),        8.82ms (100.00%, 1.58%),         519us (100.00%, 0.02%)
AddN                          1499.14MB (96.26%, 3.77%),          399us (99.75%, 0.01%),            0us (98.42%, 0.00%),          399us (99.98%, 0.01%)
ApplyAdam                       65.65MB (92.48%, 0.17%),         1.58ms (99.73%, 0.04%),            0us (98.42%, 0.00%),         1.58ms (99.97%, 0.05%)
AssignSub                       65.87MB (92.32%, 0.17%),          944us (99.69%, 0.03%),           72us (98.42%, 0.01%),          872us (99.92%, 0.03%)
BiasAdd                        144.60MB (92.15%, 0.36%),          813us (99.67%, 0.02%),          745us (98.41%, 0.13%),           68us (99.89%, 0.00%)
BiasAddGrad                      9.47KB (91.79%, 0.00%),          213us (99.64%, 0.01%),            0us (98.27%, 0.00%),          213us (99.89%, 0.01%)
Cast                           156.93KB (91.79%, 0.00%),         5.30ms (99.64%, 0.14%),           78us (98.27%, 0.01%),         5.22ms (99.88%, 0.17%)
ConcatV2                        13.38MB (91.79%, 0.03%),          335us (99.49%, 0.01%),          157us (98.26%, 0.03%),          178us (99.71%, 0.01%)
Const                           27.92KB (91.75%, 0.00%),          190us (99.48%, 0.01%),            0us (98.23%, 0.00%),          190us (99.71%, 0.01%)
Conv2D                         710.03MB (91.75%, 1.79%),       2.61sec (99.48%, 70.82%),      481.75ms (98.23%, 86.31%),       2.13sec (99.70%, 68.05%)
Conv2DBackpropFilter            75.55MB (89.97%, 0.19%),        89.35ms (28.66%, 2.43%),            0us (11.92%, 0.00%),        89.35ms (31.65%, 2.86%)
Conv2DBackpropInput           2295.51MB (89.78%, 5.78%),      556.06ms (26.23%, 15.10%),            0us (11.92%, 0.00%),      556.06ms (28.79%, 17.80%)
CropAndResize                 1742.61MB (84.00%, 4.38%),        39.63ms (11.13%, 1.08%),        28.59ms (11.92%, 5.12%),        11.04ms (10.99%, 0.35%)
CropAndResizeGradBoxes         572.93KB (79.62%, 0.00%),       155.98ms (10.06%, 4.24%),             0us (6.80%, 0.00%),       155.98ms (10.64%, 4.99%)
CropAndResizeGradImage          84.62MB (79.62%, 0.21%),        106.34ms (5.82%, 2.89%),             0us (6.80%, 0.00%),        106.34ms (5.65%, 3.40%)

```
