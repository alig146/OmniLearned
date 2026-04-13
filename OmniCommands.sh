


######### BASE COMMANDS #########################
omnilearned train --dataset tau --path /pscratch/sd/n/nkyriaco/TauCP/TauRecoML/TestAODs/processed_h5/03_26_26/ --num-feat 12 --num-classes 3 \
--use-tracks --track-dim 16 --size small --epoch 5 --output_dir checkpoints --aux-tasks-str "decay_mode:5,electron_vs_qcd:2"
omnilearned evaluate --dataset tau --path /pscratch/sd/n/nkyriaco/TauCP/TauRecoML/TestAODs/processed_h5/03_26_26/ --num-feat 12 --num-classes 3 \ 
--use-tracks --track-dim 16 --size small --input_dir ./checkpoints --output_dir results --aux-tasks-str "decay_mode:5,electron_vs_qcd:2"

############## COMMANDS WHEN USING CELL INFORMATION ####################
omnilearned train --dataset tau --path /pscratch/sd/n/nkyriaco/TauCP/TauRecoML/TestAODs/processed_h5/03_26_26/ --num-feat 12 --num-classes 3 \
--use-tracks --track-dim 16 --use-cells --cell-dim 14 \ 
--size small --epoch 5 --output_dir checkpoints --aux-tasks-str "decay_mode:5,electron_vs_qcd:2"

omnilearned evaluate --dataset tau --path /pscratch/sd/n/nkyriaco/TauCP/TauRecoML/TestAODs/processed_h5/03_26_26/ --num-feat 12 --num-classes 3 \ 
--use-tracks --track-dim 16 --use-cells --cell-dim 14 \ 
--size small --input_dir ./checkpoints --output_dir results --aux-tasks-str "decay_mode:5,electron_vs_qcd:2"



############## COMMANDS WHEN USING CELL + TauTrack Classifier Auxiliary Tasks INFORMATION ####################
omnilearned train --dataset tau --path /pscratch/sd/n/nkyriaco/TauCP/TauRecoML/TestAODs/processed_h5/03_30_26/ --num-feat 12 --num-classes 3 \
--use-tracks --track-dim 16 --use-cells --cell-dim 14 \ 
--size small --epoch 5 --output_dir checkpoints --aux-tasks-str "decay_mode:5,electron_vs_qcd:2,tautrack_class:4"

omnilearned evaluate --dataset tau --path /pscratch/sd/n/nkyriaco/TauCP/TauRecoML/TestAODs/processed_h5/03_30_26/ --num-feat 12 --num-classes 3 \ 
--use-tracks --track-dim 16 --use-cells --cell-dim 14 \ 
--size small --input_dir ./checkpoints --output_dir results --aux-tasks-str "decay_mode:5,electron_vs_qcd:2,tautrack_class:4"