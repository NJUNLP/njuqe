import click

@click.command()
@click.option("--raw-hyp", type=str, help="the path to read raw hypothesis")
@click.option("--bpe-hyp", type=str, help="the path to read bpe hypothesis")
@click.option("--raw-tag", type=str, help="the path to read raw tags")
@click.option("--bpe-tag", type=str, help="the path to write bpe tags")
def main(raw_hyp, bpe_hyp, raw_tag, bpe_tag):
    with open(raw_hyp, "r") as raw_hyp_file,\
        open(bpe_hyp, "r") as bpe_hyp_file, \
        open(raw_tag, "r") as raw_tag_file, \
        open(bpe_tag, "w") as bpe_tag_file:
        for raw_hyp_line, bpe_hyp_line, raw_tag_line in zip(raw_hyp_file, bpe_hyp_file, raw_tag_file):
            index = 0
            raw_hyp_line = raw_hyp_line.strip().split()
            bpe_hyp_line = bpe_hyp_line.strip().split()
            raw_tag_line = raw_tag_line.strip().split()
            
            assert len(raw_hyp_line) == len(raw_tag_line) - 1
            
            bpe_tags = []
            
            for word, tag in zip(raw_hyp_line, raw_tag_line):
                merge_bpe_word =""
                while index < len(bpe_hyp_line) and merge_bpe_word != word:
                    if bpe_hyp_line[index].endswith("@@"):
                        merge_bpe_word += bpe_hyp_line[index][:-2]
                    else:
                        merge_bpe_word += bpe_hyp_line[index]
                    index += 1
                    bpe_tags.append(tag)
                assert merge_bpe_word == word, f"merge_bpe_word: {merge_bpe_word}, word: {word}"
            bpe_tags.append(raw_tag_line[-1])
            bpe_tag_file.write(" ".join(bpe_tags) + "\n")
    
    
if __name__ == "__main__":
    main()