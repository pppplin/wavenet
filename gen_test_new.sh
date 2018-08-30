# #!/bin/bashfor ((i=0; i<10; i+=1)); do
# for ((i=0; i<10; i+=1)); do
#     #python generate.py ./logdir/Nottingham/train/2018-08-21T22-44-20/model.ckpt-12830 --samples 840 --wav_seed ./Nottingham_melody_64_hashed/mel10/hpps_41.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/gc64_all_hpps_41_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 64 --fast_generation True
#     #512
#     #python generate.py ./logdir/Nottingham/train/2018-08-13T18-52-50/model.ckpt-11950 --samples 840 --wav_seed ./Nottingham_melody_64_hashed/mel10/hpps_41.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/512_all_hpps_41_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 32 --fast_generation True
#     #all reg
#     python generate.py ./logdir/Nottingham/train/2018-08-13T22-51-21/model.ckpt-8999 --samples 840 --wav_seed ./Nottingham_melody_64_hashed/mel10/hpps_41.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/reg_all_hpps_41_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 32 --fast_generation True
#     #all nc
#     python generate.py ./logdir/Nottingham/train/2018-08-28T05-48-34/model.ckpt-9600 --samples 840 --wav_seed ./Nottingham_melody_64_hashed/mel10/hpps_41.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/nc_all_hpps_41_${i}.mid --load_chord False --fast_generation True
#     #half reg
#     python generate.py ./logdir/Nottingham/train/2018-08-13T22-51-21/model.ckpt-8999 --samples 487 --wav_seed ./Nottingham_melody_64_hashed/mel10/hpps_41_half.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/reg_half_hpps_41_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 32 --fast_generation True
#     #half nc
#     python generate.py ./logdir/Nottingham/train/2018-08-28T05-48-34/model.ckpt-9600 --samples 487 --wav_seed ./Nottingham_melody_64_hashed/mel10/hpps_41_half.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/nc_half_hpps_41_${i}.mid --load_chord False --fast_generation True
#     #half gc64
#     python generate.py ./logdir/Nottingham/train/2018-08-21T22-44-20/model.ckpt-12830 --samples 487 --wav_seed ./Nottingham_melody_64_hashed/mel10/hpps_41_half.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/gc64_half_hpps_41_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 64 --fast_generation True
# done

# for ((i=0; i<10; i+=1)); do
#     #all reg
#     python generate.py ./logdir/Nottingham/train/2018-08-13T22-51-21/model.ckpt-8999 --samples 624 --wav_seed ./Nottingham_melody_64_hashed/mel10/ashover_40.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/reg_all_ashover_40_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 32 --fast_generation True
#     #all nc
#     python generate.py ./logdir/Nottingham/train/2018-08-28T05-48-34/model.ckpt-9600 --samples 624 --wav_seed ./Nottingham_melody_64_hashed/mel10/ashover_40.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/nc_all_ashover_40_${i}.mid --load_chord False --fast_generation True
#     #half reg
#     python generate.py ./logdir/Nottingham/train/2018-08-13T22-51-21/model.ckpt-8999 --samples 385 --wav_seed ./Nottingham_melody_64_hashed/mel10/ashover_40_half.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/reg_half_ashover_40_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 32 --fast_generation True
#     #half nc
#     python generate.py ./logdir/Nottingham/train/2018-08-28T05-48-34/model.ckpt-9600 --samples 385 --wav_seed ./Nottingham_melody_64_hashed/mel10/ashover_40_half.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/nc_half_ashover_40_${i}.mid --load_chord False --fast_generation True
#     #half gc64
#     python generate.py ./logdir/Nottingham/train/2018-08-21T22-44-20/model.ckpt-12830 --samples 385 --wav_seed ./Nottingham_melody_64_hashed/mel10/ashover_40_half.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/gc64_half_ashover_40_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 64 --fast_generation True
# done

for ((i=0; i<15; i+=1)); do
    #all reg
    #python generate.py ./logdir/Nottingham/train/2018-08-13T22-51-21/model.ckpt-8999 --samples 3496 --wav_seed ./Nottingham_melody_64_hashed/mel10/morris_11.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/reg_all_morris_11_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 32 --fast_generation True
    #all nc
    #python generate.py ./logdir/Nottingham/train/2018-08-28T05-48-34/model.ckpt-9600 --samples 3496 --wav_seed ./Nottingham_melody_64_hashed/mel10/morris_11.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/nc_all_morris_11_${i}.mid --load_chord False --fast_generation True
    #half reg
    #python generate.py ./logdir/Nottingham/train/2018-08-13T22-51-21/model.ckpt-8999 --samples 1815 --wav_seed ./Nottingham_melody_64_hashed/mel10/morris_11_half.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/reg_half_morris_11_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 32 --fast_generation True
    #half nc
    #python generate.py ./logdir/Nottingham/train/2018-08-28T05-48-34/model.ckpt-9600 --samples 1815 --wav_seed ./Nottingham_melody_64_hashed/mel10/morris_11_half.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/nc_half_morris_11_${i}.mid --load_chord False --fast_generation True
    #half gc64
    #python generate.py ./logdir/Nottingham/train/2018-08-21T22-44-20/model.ckpt-12830 --samples 1815 --wav_seed ./Nottingham_melody_64_hashed/mel10/morris_11_half.mid --mid_out_path ./Nottingham_melody_64_hashed/mel10_generated/gc64_half_morris_11_${i}.mid --load_chord True --gc_cardinality 52 --gc_channels 64 --fast_generation True
    #512
    #nc_all
    python generate.py ./logdir/Nottingham/train/2018-08-28T05-48-34/model.ckpt-9600 --samples 624 --wav_seed ./Nottingham_melody_64_hashed/mel10/ashover_40.mid --mid_out_path ./Nottingham_melody_64_hashed/nc_all/nc_all_ashover_40_${i}.mid --load_chord False --fast_generation True
    python generate.py ./logdir/Nottingham/train/2018-08-28T05-48-34/model.ckpt-9600 --samples 840 --wav_seed ./Nottingham_melody_64_hashed/mel10/hpps_41.mid --mid_out_path ./Nottingham_melody_64_hashed/nc_all/nc_all_hpps_41_${i}.mid --load_chord False --fast_generation True
    #nc_all morris11
    #python generate.py ./logdir/Nottingham/train/2018-08-28T05-48-34/model.ckpt-9600 --samples 3496 --wav_seed ./Nottingham_melody_64_hashed/mel10/morris_11.mid --mid_out_path ./Nottingham_melody_64_hashed/morris_temp/nc_all_morris_11_${i}.mid --load_chord False --fast_generation True
done



