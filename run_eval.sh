source "/mnt/bn/jliang-lq-nas/workplace/init_workplace.sh"
conda activate eval_gen
# enable_proxy
# export http_proxy=http://sys-proxy-rd-relay.byted.org:8118  https_proxy=http://sys-proxy-rd-relay.byted.org:8118  no_proxy=code.byted.org

python3 tage \
    --target-audio-dir "/mnt/bn/jliang-lq-nas/workplace/AudioSet-E/dataset/add/mini_mini_data" \
    --generated-audio-dir "/mnt/bn/jliang-lq-nas/workplace/AudioSet-E/dataset/add/mini_mini_data" \
    --reference_text_path "/mnt/bn/jliang-lq-nas/workplace/AudioSet-E/dataset/add/val-gt-without_win.json"
    # --recalculate