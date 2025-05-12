#!/bin/bash
#$ -l h_rt=1:0:0
##$ -l gpu_type='ampere'
##$ -l node_type='rdg'
#$ -l gpu=1
#$ -l h_vmem=7.5G
#$ -pe smp 8
#$ -wd /data/home/eey340/WORKPLACE/TAGE
#$ -j y
#$ -N eval-remove
#$ -o /data/scratch/eey340/am-FINAL/remove/eval-Text-noTan.log
#$ -m beas

##############
##   Please read documentation to find out how to create GPU job:
##   https://docs.hpc.qmul.ac.uk/using/usingGPU/
###############

# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------
source /data/home/eey340/venvs/tage/bin/activate
task=remove

# trials=("Tango-noTan-withText" "Tango-Tan-withText" "Tango-noTan-withoutText" "Tango-Tan-withoutText" "Tango-noTan-withText-25" "Tango-noTan-withText-100")

# for t in "${trials[@]}"; do
#     python3 tage \
#         --target-audio-dir "/data/EECS-MachineListeningLab/datasets/AudioSet-E/dataset/${task}/target" \
#         --generated-audio-dir "/data/scratch/eey340/am/outputs-tango2/${task}/${t}" \
#         # --reference_text_path "/mnt/bn/jliang-lq-nas/workplace/AudioSet-E/dataset/add/val-gt-without_win.json"
#         # --recalculate
# done
# 
# task=("add" "remove" "replace")
# for t in "${task[@]}"; do
#     python3 tage \
#         --target-audio-dir "/data/EECS-MachineListeningLab/datasets/AudioSet-E/dataset/${t}/target" \
#         --generated-audio-dir "/data/scratch/eey340/data_package/ddim_inv/${t}" \
#         # --reference_text_path "/mnt/bn/jliang-lq-nas/workplace/AudioSet-E/dataset/add/val-gt-without_win.json"
#         # --recalculate
# done

# python3 tage \
#     --target-audio-dir "/data/EECS-MachineListeningLab/datasets/AudioSet-E/dataset/add/target" \
#     --generated-audio-dir "/data/scratch/eey340/data_package/SoundEdit-TrainingFree/add/NEW_FINAL-FINAL_EVAL-noTan-withText" \
#     # --reference_text_path "/mnt/bn/jliang-lq-nas/workplace/AudioSet-E/dataset/add/val-gt-without_win.json"
#     # --recalculate


# w_contents=(20 40 60)
# w_edits=(0.05 0.1 0.2)
# w_contrasts=(0.25 0.5 1.0)
# guidance_scales=(2.4 4.8 6)

# for w_content in "${w_contents[@]}"
# do
#     for w_edit in "${w_edits[@]}"
#     do
#         for w_contrast in "${w_contrasts[@]}"
#         do
#             for guidance_scale in "${guidance_scales[@]}"
#             do
#             python3 tage \
#                 --target-audio-dir "/data/scratch/eey340/AudioSet-E/remove/mini_data" \
#                 --generated-audio-dir "/data/scratch/eey340/am/outputs-tango2/remove/NEW-Tango2-w_cont${w_content}-w_e${w_edit}-w_const${w_contrast}-gs${guidance_scale}"
#             done
#         done
#     done
# done


# model_name=("audioldm" "tango" "tango2-full") # "cvssp/audioldm-l-full" | "declare-lab/tango" | "declare-lab/tango2-full"
# mode="noText-noTan"  # noText-noTan | noText-Tan | Text-noTan | Text-Tan
# w_contents=(5 10 15)
# w_edits=(10 15 20)
# guidance_scales=(0.6 1.2 1.8)
# for mdl in "${model_name[@]}"
# do
#     for w_content in "${w_contents[@]}"
#     do
#         for w_edit in "${w_edits[@]}"
#         do
#             for guidance_scale in "${guidance_scales[@]}"
#             do
#             outpath="/data/scratch/eey340/am-FINAL/add/${mdl}/${mode}/w_cont${w_content}-w_e${w_edit}-w_const-gs${guidance_scale}"
#             python3 tage \
#                 --target-audio-dir "/data/scratch/eey340/AudioSet-E/add/mini_data" \
#                 --generated-audio-dir ${outpath} \
#                 --recalculate
#             done
#         done
#     done
# done


# model_name=("audioldm" "tango" "tango2-full") # "cvssp/audioldm-l-full" | "declare-lab/tango" | "declare-lab/tango2-full"
# mode="noText-Tan"  # noText-noTan | noText-Tan | Text-noTan | Text-Tan
# w_contents=(5 10 15)
# w_edits=(10 15 20)
# guidance_scales=(1.2 2.4 3.6)
# for mdl in "${model_name[@]}"
# do
#     for w_content in "${w_contents[@]}"
#     do
#         for w_edit in "${w_edits[@]}"
#         do
#             for guidance_scale in "${guidance_scales[@]}"
#             do
#             outpath="/data/scratch/eey340/am-FINAL/add/${mdl}/${mode}/w_cont${w_content}-w_e${w_edit}-w_const-gs${guidance_scale}"
#             python3 tage \
#                 --target-audio-dir "/data/scratch/eey340/AudioSet-E/add/mini_data" \
#                 --generated-audio-dir ${outpath}
#             done
#         done
#     done
# done


# model_name=("audioldm" "tango" "tango2-full") # "cvssp/audioldm-l-full" | "declare-lab/tango" | "declare-lab/tango2-full"
# mode="Text-noTan"  # noText-noTan | noText-Tan | Text-noTan | Text-Tan
# w_contents=(5 10 15)
# w_edits=(10 15 20)
# guidance_scales=(0.6 1.2 1.8)
# for mdl in "${model_name[@]}"
# do
#     for w_content in "${w_contents[@]}"
#     do
#         for w_edit in "${w_edits[@]}"
#         do
#             for guidance_scale in "${guidance_scales[@]}"
#             do
#             outpath="/data/scratch/eey340/am-FINAL/add/${mdl}/${mode}/w_cont${w_content}-w_e${w_edit}-w_const-gs${guidance_scale}"
#             python3 tage \
#                 --target-audio-dir "/data/scratch/eey340/AudioSet-E/add/mini_data" \
#                 --generated-audio-dir ${outpath}
#                 --recalculate
#             done
#         done
#     done
# done


# model_name=("audioldm" "tango" "tango2-full") # "cvssp/audioldm-l-full" | "declare-lab/tango" | "declare-lab/tango2-full"
# mode="Text-Tan"  # noText-noTan | noText-Tan | Text-noTan | Text-Tan
# w_contents=(5 10 15)
# w_edits=(10 15 20)
# guidance_scales=(1.2 2.4 3.6)
# for mdl in "${model_name[@]}"
# do
#     for w_content in "${w_contents[@]}"
#     do
#         for w_edit in "${w_edits[@]}"
#         do
#             for guidance_scale in "${guidance_scales[@]}"
#             do
#             outpath="/data/scratch/eey340/am-FINAL/add/${mdl}/${mode}/w_cont${w_content}-w_e${w_edit}-w_const-gs${guidance_scale}"
#             python3 tage \
#                 --target-audio-dir "/data/scratch/eey340/AudioSet-E/add/mini_data" \
#                 --generated-audio-dir ${outpath}
#             done
#         done
#     done
# done


# model_name=("audioldm" "tango" "tango2-full") # "cvssp/audioldm-l-full" | "declare-lab/tango" | "declare-lab/tango2-full"
# mode="Text-noTan"  # noText-noTan | noText-Tan | Text-noTan | Text-Tan
# w_contents=(5 10 15)
# w_edits=(10 15 20)
# guidance_scales=(0.6 1.2 1.8)
# for mdl in "${model_name[@]}"
# do
#     for w_content in "${w_contents[@]}"
#     do
#         for w_edit in "${w_edits[@]}"
#         do
#             for guidance_scale in "${guidance_scales[@]}"
#             do
#             outpath="/data/scratch/eey340/am-FINAL/add/${mdl}/${mode}/w_cont${w_content}-w_e${w_edit}-w_const-gs${guidance_scale}"
#             python3 tage \
#                 --target-audio-dir "/data/scratch/eey340/AudioSet-E/add/mini_data" \
#                 --generated-audio-dir ${outpath}
#                 --recalculate
#             done
#         done
#     done
# done


# model_name=("audioldm" "tango" "tango2-full") # "cvssp/audioldm-l-full" | "declare-lab/tango" | "declare-lab/tango2-full"
# mode="Text-noTan"  # noText-noTan | noText-Tan | Text-noTan | Text-Tan
# w_contents=(20 40 60)
# w_edits=(0.1 0.2 0.3)
# w_contrasts=(0.25 0.5 0.75)
# guidance_scales=(2.4 4.8 7.2)
# for mdl in "${model_name[@]}"
# do
#     for w_content in "${w_contents[@]}"
#     do
#         for w_edit in "${w_edits[@]}"
#         do
#             for w_contrast in "${w_contrasts[@]}"
#                 do
#                 for guidance_scale in "${guidance_scales[@]}"
#                 do
#                 outpath="/data/scratch/eey340/am-FINAL/remove/${mdl}/${mode}/w_cont${w_content}-w_e${w_edit}-w_const${w_contrast}-gs${guidance_scale}"
#                 python3 tage \
#                     --target-audio-dir "/data/scratch/eey340/AudioSet-E/remove/mini_data" \
#                     --generated-audio-dir ${outpath} \
#                     --recalculate
#                 done
#             done
#         done
#     done
# done


python3 tage \
    --target-audio-dir "/data/EECS-MachineListeningLab/datasets/AudioSet-E/dataset/add/target" \
    --generated-audio-dir "/data/scratch/eey340/data_package/ddim_inv/add/" \
    # --reference_text_path "/mnt/bn/jliang-lq-nas/workplace/AudioSet-E/dataset/add/val-gt-without_win.json"
    # --recalculate