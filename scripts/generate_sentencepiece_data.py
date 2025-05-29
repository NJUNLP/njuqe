import sentencepiece as spm
import numpy as np
import os
import click
from tqdm.contrib.concurrent import process_map
import subprocess

sp = spm.SentencePieceProcessor()
sp.Load('')
fake = False


def process(args):

    line = args

    token_pieces = ""
    bounds = ""
    offset = 0
    tokens = line.strip().split()

    result = {}

    if fake:
        fake_tags = ""
        for _ in range(len(tokens)):
            if np.random.random() < 0.5:
                fake_tags += "OK "
            else:
                fake_tags += "BAD "
        result['fake_tags'] = fake_tags[:-1] + "\n"
        result['fake_score'] = str(np.random.random()) + "\n"

    for token in tokens:
        pieces = sp.EncodeAsPieces(token)
        for piece in pieces:
            token_pieces += "".join(piece) + " "
        bounds += str(offset) + " "
        offset += len(pieces)

        result["tokens"] = token_pieces
        result["bounds"] = bounds

    return result


def generate_data(file_path, output_path, bounds_path=None):
    sp = spm.SentencePieceProcessor()
    sp.Load('/home/nfs03/laizj/code/util/sentencepiece/sentencepiece.bpe.model')

    try:
        result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, check=True)

        # 获取命令的标准输出并打印行数
        output = result.stdout
        line_count = int(output.split()[0])
        print(f'文件"{file_path}"的行数为: {line_count}')
    except subprocess.CalledProcessError as e:
        print(f'执行命令时出错: {e}')

    with open(file_path) as f:
        answers = process_map(process, f.readlines(),
                              total=line_count, max_workers=32, chunksize=50)

    with open(output_path, "w") as o:
        for ans in answers:
            o.write(ans['tokens'].strip() + "\n")

    if bounds_path:
        with open(bounds_path, "w") as b:
            for ans in answers:
                b.write(ans['bounds'].strip() + "\n")

    if fake:
        with open(os.path.dirname(output_path) + "/fake.tag", "w") as fake_tag_f, open(os.path.dirname(output_path) + "/fake.score", "w") as fake_score_f:
            for ans in answers:
                fake_tag_f.write(ans['fake_tags'])
                fake_score_f.write(ans['fake_score'])


@click.command()
@click.option('--input-dir', required=True, type=click.Path(exists=True), help="path with out /")
@click.option('--output-dir', required=True, type=click.Path(exists=True), help="path with out /")
@click.option('--src', required=True, type=str)
@click.option('--mt', required=True, type=str)
@click.option('--train', default=False, is_flag=True, type=bool)
@click.option('--valid', default=False, is_flag=True, type=bool)
@click.option('--test', default=False, is_flag=True, type=bool)
@click.option('--no-need-bounds', default=False, is_flag=True, type=bool, help="Not to give bounds")
def main(input_dir, output_dir, src, mt, train, valid, test, no_need_bounds):
    """ auto process {split}.{src}-{mt}.{src}/{mt} files to encode into spm"""

    input_dir += "/"
    output_dir += "/"

    # mt
    if train:
        train_mt = input_dir + f"train.{src}-{mt}.{mt}"
        train_mt_output = output_dir + f"train.{src}-{mt}.spm.{mt}"
        if no_need_bounds:
            generate_data(train_mt, train_mt_output)
        else:
            train_mt_bounds = output_dir + f"train.{mt}.bounds"
            generate_data(train_mt, train_mt_output, train_mt_bounds)

    if valid:
        valid_mt = input_dir + f"valid.{src}-{mt}.{mt}"
        valid_mt_output = output_dir + f"valid.{src}-{mt}.spm.{mt}"
        if no_need_bounds:
            generate_data(valid_mt, valid_mt_output)
        else:
            valid_mt_bounds = output_dir + f"valid.{mt}.bounds"
            generate_data(valid_mt, valid_mt_output, valid_mt_bounds)

    if test:
        test_mt = input_dir + f"test.{src}-{mt}.{mt}"
        test_mt_output = output_dir + f"test.{src}-{mt}.spm.{mt}"
        if no_need_bounds:
            generate_data(test_mt, test_mt_output)
        else:
            test_mt_bounds = output_dir + f"test.{mt}.bounds"
            generate_data(test_mt, test_mt_output, test_mt_bounds)

    # src
    if train:
        train_src = input_dir + f"train.{src}-{mt}.{src}"
        train_src_output = output_dir + f"train.{src}-{mt}.spm.{src}"
        if no_need_bounds:
            generate_data(train_src, train_src_output)
        else:
            train_src_bounds = output_dir + f"train.{src}.bounds"
            generate_data(train_src, train_src_output, train_src_bounds)

    if valid:
        valid_src = input_dir + f"valid.{src}-{mt}.{src}"
        valid_src_output = output_dir + f"valid.{src}-{mt}.spm.{src}"
        if no_need_bounds:
            generate_data(valid_src, valid_src_output)
        else:
            valid_src_bounds = output_dir + f"valid.{src}.bounds"
            generate_data(valid_src, valid_src_output, valid_src_bounds)

    if test:
        test_src = input_dir + f"test.{src}-{mt}.{src}"
        test_src_output = output_dir + f"test.{src}-{mt}.spm.{src}"
        if no_need_bounds:
            generate_data(test_src, test_src_output)
        else:
            test_src_bounds = output_dir + f"test.{src}.bounds"
            generate_data(test_src, test_src_output, test_src_bounds)

    print("all finiszhd")


if __name__ == '__main__':
    main()
