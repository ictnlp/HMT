export CUDA_VISIBLE_DEVICES=0
MODELFILE=dir_to_save_model
DATAFILE=dir_to_data
REF=path_to_reference

# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${MODELFILE} --num-update-checkpoints 5 --output ${MODELFILE}/average-model.pt 

# generate translation
python generate.py ${DATAFILE} --path ${MODELFILE}/average-model.pt --batch-size 1 --beam 1 --left-pad-source False --fp16  --remove-bpe --sim-decoding > pred.out

grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
multi-bleu.perl -lc ${REF} < pred.translation