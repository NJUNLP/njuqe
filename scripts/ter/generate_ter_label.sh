# THis is the script to use tercom toolkit.

NJUQE_PATH=#your njuqe path

# preprocess your output ref/hyp file
python $NJUQE_PATH/scripts/ter/convert_file.py FILE_PATH/ref.txt FILE_PATH/ref_standard.txt
python $NJUQE_PATH/scripts/ter/convert_file.py FILE_PATH/hyp.txt FILE_PATH/hyp_standard.txt

# tercom.7.25.jar is from https://www.cs.umd.edu/~snover/tercom/ and get outputs
# NOTE: when generate tags file, you can disabling shifts by using the `-d 0` option
java -jar tercom.7.25.jar -r FILE_PATH/ref_standard.txt -h FILE_PATH/hyp_standard.txt -n OUT_PATH

# generate your tags and hter file
python $NJUQE_PATH/scripts/ter/convert_label.py OUT_PATH/out.pra \
    OUT_PATH/tags.txt \
    OUT_PATH/hter.txt