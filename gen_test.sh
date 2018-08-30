#!/bin/bashfor ((i=0; i<10; i+=1)); do
for ((i=1; i<10; i+=1)); do
    #python generate.py ./logdir/Nottingham/train/2018-08-28T05-48-34/model.ckpt-9600 --samples 160 --wav_seed ./Nottingham_melody_64_hashed/mel10/morris_11.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/nc_morris_11_${i}.mid --load_chord False --fast_generation True
    python generate.py ./logdir/Nottingham/train/2018-08-21T22-44-20/model.ckpt-12830 --samples 840 --wav_seed ./Nottingham_melody_64_hashed/mel10/hpps_41.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/gc64_all_hpps_41_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 64 --fast_generation True
    #python generate.py ./logdir/Nottingham/train/2018-08-13T18-52-50/model.ckpt-11950 --samples 840 --wav_seed ./Nottingham_melody_64_hashed/mel10/hpps_41.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/512_all_hpps_41_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 32 --fast_generation True
done

for ((i=1; i<10; i+=1)); do
    python generate.py ./logdir/Nottingham/train/2018-08-21T22-44-20/model.ckpt-12830 --samples 3496 --wav_seed ./Nottingham_melody_64_hashed/mel10/morris_11.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/gc64_all_morris_11_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 64 --fast_generation True
    #python generate.py ./logdir/Nottingham/train/2018-08-13T18-52-50/model.ckpt-11950 --samples 3496 --wav_seed ./Nottingham_melody_64_hashed/mel10/morris_11.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/512_all_morris_11_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 32 --fast_generation True
done

for ((i=1; i<10; i+=1)); do
    python generate.py ./logdir/Nottingham/train/2018-08-21T22-44-20/model.ckpt-12830 --samples 624 --wav_seed ./Nottingham_melody_64_hashed/mel10/ashover_40.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/gc64_all_ashover_40_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 64 --fast_generation True
    #python generate.py ./logdir/Nottingham/train/2018-08-13T18-52-50/model.ckpt-11950 --samples 624 --wav_seed ./Nottingham_melody_64_hashed/mel10/ashover_40.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/512_all_ashover_40_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 32 --fast_generation True
done


