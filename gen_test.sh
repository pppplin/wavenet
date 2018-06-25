#!/bin/bash
for ((i=59; i<66; i+=6)); do
    #python generate.py ./logdir/Nottingham/train/2018-06-18T20-56-57/model.ckpt-19999 --samples 480 --load_velocity False --fast_generation True --wav_seed ./Nottingham_melody_64_hashed/test_half/ashover_simple_chords_${i}.mid --mid_out_path ./Nottingham_melody_64_hashed/generated/16-480-global-temp-ashover_simple_chords_${i}.mid
    python generate.py ./logdir/Nottingham/train/2018-06-20T20-43-23/model.ckpt-2999 --samples 320 --wav_seed ./Nottingham_melody_64_hashed/test_cut_melody/hpps_simple_chords_${i}.mid --mid_out_path ./Nottingham_melody_64_hashed/generated/16-320-global-hpps_simple_chords_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 64 --fast_generation True
done

#python generate.py ./logdir/ssccm/train/2018-05-08T22-21-05/model.ckpt-3200 --samples 480 --load_velocity False --wav_seed ./ssccm/test/ssccm351-mel.mid --mid_out_path ./ssccm/final/ssccm351-mel-8-480-relu-relu-2018-05-08T22-21-05.mid --fast_generation True
