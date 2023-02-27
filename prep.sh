SRC=source_language
TGT=target_language
TRAIN=path_to_train_data
VAIILD=path_to_vaild_data
TEST=path_to_test_data
DATAFILE=dir_to_data

# add --joined-dictionary for WMT15 German-English
fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT} \
    --trainpref ${TRAIN} --validpref ${VAIILD} \
    --testpref ${TEST}\
    --destdir ${DATAFILE} \
    --workers 20