import random
import click
import os
from tqdm import tqdm


def read_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return file.readlines()
    return None


def write_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as file:
        for line in data:
            file.write(line)


def group_data(group_sizes, data):
    if data is None:
        return None
    grouped_data = []
    index = 0
    for count in group_sizes:
        grouped_data.append(data[index:index + count])
        index += count
    return grouped_data


def split_data(grouped_data, valid_indices):
    if grouped_data is None:
        return None, None
    train, valid = [], []
    index = 0
    le = len(valid_indices)
    for i, item in enumerate(grouped_data):
        if index < le and i == valid_indices[index]:
            valid.extend(item)
            index += 1
        else:
            train.extend(item)

    return train, valid


@click.command()
@click.option('--input-dir', required=True, type=str)
@click.option('--output-dir', required=True, type=str)
@click.option('--src', required=True, type=str)
@click.option('--mt', required=True, type=str)
def main(input_dir, output_dir, src, mt):

    # Read the contains file to get the grouping information
    contains = read_file(input_dir + '/train.raw.contains')

    # Read other files
    text_src = read_file(input_dir + f'/train.raw.{src}-{mt}.{src}')
    text_mt = read_file(input_dir + f'/train.raw.{src}-{mt}.{mt}')
    errors = read_file(input_dir + '/train.raw.errors')
    mqm_weight = read_file(input_dir + '/train.raw.mqm_score')
    dtag = read_file(input_dir + '/train.raw.dtag')
    tag = read_file(input_dir + '/train.raw.tag')
    
    # Group sizes
    if contains:
        group_sizes = [int(x.strip()) for x in contains]
    else:
        group_sizes = [1] * len(text_src)

    # Group data
    grouped_src = group_data(group_sizes, text_src)
    grouped_mt = group_data(group_sizes, text_mt)
    grouped_errors = group_data(group_sizes, errors)
    grouped_mqm_weight = group_data(group_sizes, mqm_weight)
    grouped_dtag = group_data(group_sizes, dtag)
    grouped_tag = group_data(group_sizes, tag)

    # Select indices for valid groups
    # Change this number as needed
    num_valid_groups = max(min(1000, len(group_sizes)),
                           int(len(group_sizes) * 0.002))
    valid_indices = sorted(random.sample(
        range(len(group_sizes)), num_valid_groups))
    train_indices = random.sample(range(len(group_sizes)), len(group_sizes))

    # Split data into train and valid sets
    train_contains, valid_contains = split_data(contains, valid_indices)
    train_errors, valid_errors = split_data(grouped_errors, valid_indices)
    train_src, valid_src = split_data(grouped_src, valid_indices)
    train_mt, valid_mt = split_data(grouped_mt, valid_indices)
    train_mqm_weight, valid_mqm_weight = split_data(
        grouped_mqm_weight, valid_indices)
    train_dtag, valid_dtag = split_data(grouped_dtag, valid_indices)
    train_tag, valid_tag = split_data(grouped_tag, valid_indices)

    # Write valid data to new files
    if train_contains:
        write_file(output_dir + '/train.contains', train_contains)
    if train_errors:
        write_file(output_dir + '/train.errors', train_errors)
    write_file(output_dir + f'/train.{src}-{mt}.{src}', train_src)
    write_file(output_dir + f'/train.{src}-{mt}.{mt}', train_mt)
    write_file(output_dir + '/train.mqm_score', train_mqm_weight)
    if train_dtag:
        write_file(output_dir + '/train.dtag', train_dtag)
    write_file(output_dir + '/train.tag', train_tag)

    if valid_contains:
        write_file(output_dir + '/valid.contains', valid_contains)
    if valid_errors:
        write_file(output_dir + '/valid.errors', valid_errors)
    write_file(output_dir + f'/valid.{src}-{mt}.{src}', valid_src)
    write_file(output_dir + f'/valid.{src}-{mt}.{mt}', valid_mt)
    write_file(output_dir + '/valid.mqm_score', valid_mqm_weight)
    if valid_dtag:
        write_file(output_dir + '/valid.dtag', valid_dtag)
    write_file(output_dir + '/valid.tag', valid_tag)


if __name__ == "__main__":
    main()
