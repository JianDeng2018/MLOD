# Demos

1. Make sure `mlod` is installed -> `python setup.py install` and run the demos from the mlod top dir :

``` bash
# From top level mlod folder
python demos/show_predictions_3d.py
```

2. Some demos require `VTK-7.1.1` to be installed. For instructions on how to install see `wavedata/scripts/install/vtk_install.bash`
3. Most of the demos require an existing checkpoint and already evaluated predictions. Here's an example:
    - checkpoint\_name = 'mlod\_exp\_example'
    - data\_split = 'val'
    - global\_step = 100000

So in this case, the checkpoint\_name is the name of the experiment you ran, data\_split is the split at which that checkpoint was evaluated and global\_step is the step at which you evaluated the split.

4. Some demos only show the predictions for a given sample. Make sure the sample you are selecting, is drawn from the same evaluation split. This is with the exception for `show_predictions_2d.py` where it just generates the results over the entire validation set and stores the final images.

5. Most demos have configuration options inside their `main`. Just look for the `# Options` indicator.

6. Demos including `box_4c_vis.py` and `box_8c_vis.py` require also the predicted corners, that is before getting converted to `box_3d` format. This is already being stored when you train with `box_8c` or `box_4c` variations of box representations.

