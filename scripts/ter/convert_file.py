import sys

def func(input_file, output_file):
    with open(input_file) as f:
        input_lines = f.readlines()
    output_lines = []
    i = 1
    for line in input_lines:
        line = line.strip("\n") + " (" + str(i) + ")" + "\n"
        i = i + 1
        output_lines.append(line)
    with open(output_file, "w") as df:
        df.writelines(output_lines)


if __name__ == '__main__':

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    func(input_file, output_file)
    print('Over')