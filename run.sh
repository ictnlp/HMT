export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODELFILE=dir_to_save_model
DATAFILE=dir_to_data

FIRST_READ=3
CANDS_PER_TOKEN=6

python train.py --ddp-backend=no_c10d ${DATAFILE} --arch transformer --share-all-embeddings \
 --optimizer adam \
 --adam-betas '(0.9, 0.98)' \
 --clip-norm 0.0 \
 --lr 5e-4 \
 --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 \
 --warmup-updates 4000 \
 --dropout 0.3 \
 --encoder-attention-heads 8 \
 --decoder-attention-heads 8 \
 --criterion label_smoothed_cross_entropy \
 --label-smoothing 0.1 \
 --left-pad-source False \
 --save-dir ${MODELFILE} \
 --first-read ${FIRST_READ} \
 --cands-per-token ${CANDS_PER_TOKEN} \
 --max-tokens 4096 --update-freq 1 \
 --max-target-positions 200 \
 --skip-invalid-size-inputs-valid-test \
 --fp16 \
 --save-interval-updates 1000 \
 --keep-interval-updates 300 \
 --log-interval 10