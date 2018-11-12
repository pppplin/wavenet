# A TensorFlow implementation of DeepMind's WaveNet paper

This is the document for some newly added features.

## Train

```bash
python train.py --data_dir=corpus
```
to train the network, where `corpus` is a directory containing `.wav` files.
The script will recursively collect all `.wav` files in the directory.

You can see documentation on each of the training settings by running
```bash
python train.py --logdir_root root --data_dir data --load_chord True --gc_cardinality 52 --gc_channels 32 --chain_mel --sample size --condition_restriction
```
--load_chord: chord as global condition (should have same length with melody)
--load_velocity: similar to chord. velocity as condition.
--gc_cardinality: number of types of global conditions. For chords, the number of chords. 
--gc_channels: number of global condition channels.
--chain_mel and --chain_vel: for gibbs sampling.
--condition_restriction: only add chord conditions for the number of layers before the restriction.
--lc_channels: local condition channels. (Not guaranteed to work!!) 
Some other constraints:

You can find the configuration of the model parameters in [`wavenet_params.json`](./wavenet_params.json).
These need to stay the same between training and generation.

### Generation
```
python generate.py log_path --samples N --wav_seed wav_seed --mid_out_path mid_out_path --load_chord True --gc_cardinality 52 --gc_channels 32 --fast_generation True
```

## Visualization with Tensorboard
```
tensorboard --logdir logdir --port port
```


