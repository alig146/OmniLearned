omnilearned evaluate --dataset tau --path /pscratch/sd/n/nkyriaco/TauCP/TauRecoML/TestAODs/processed_h5/04_10_26/  \
   --num-feat 12 --num-classes 3 --use-tracks --track-dim 16 \
   --size small --input_dir checkpoints \
   --output_dir results \
   --do-vertex-classification \
   --aux-tasks-str="decay_mode:5,tes:1,tau_eta:1,tau_phi:1,charged_pion_pt:1,charged_pion_eta:1,charged_pion_phi:1,neutral_pion_pt:1,neutral_pion_eta:1,neutral_pion_phi:1" \
   --aux-regression-tasks-str="tes,tau_eta,tau_phi,charged_pion_pt,charged_pion_eta,charged_pion_phi,neutral_pion_pt,neutral_pion_eta,neutral_pion_phi"