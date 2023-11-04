# coding=utf-8

import sys

def func(output_file, fake_label, fake_score):
    with open(output_file) as of:
        o_lines = of.readlines()

    fake_label_ctxt = []
    fake_score_ctxt = []

    for line in o_lines:
        line_words = line.split()
        if len(line_words) > 0:
            if line_words[0] == "Alignment:":
                new_line = line[12:-2].replace(" ", "o")
                tags_tmp = ""
                for word in new_line:
                    if word == "o":
                        tags_tmp += "OK "
                    elif word == "D":
                        continue
                    else:
                        tags_tmp += "BAD "
                fake_label_ctxt.append(tags_tmp.strip()+"\n")
            elif line_words[0] == "Score:":
                fake_score_ctxt.append(line_words[1]+"\n")

    with open(fake_label, "w") as df:
        df.writelines(fake_label_ctxt)
    with open(fake_score, "w") as df:
        df.writelines(fake_score_ctxt)


if __name__ == '__main__':

    output_file = sys.argv[1]
    fake_label = sys.argv[2]
    fake_score = sys.argv[3]
    func(output_file, fake_label, fake_score)

    print('Over')