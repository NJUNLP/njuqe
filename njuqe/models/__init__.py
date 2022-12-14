# Copyright (c) 2020 the NJUQE authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import os

from fairseq.models import MODEL_REGISTRY, ARCH_MODEL_INV_REGISTRY
from .direct_qe import DirectQE


__all__ = [
    "DirectQE",
]

# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("njuqe.models." + model_name)

        # extra `model_parser` for sphinx
        if model_name in MODEL_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_archs = parser.add_argument_group("Named architectures")
            group_archs.add_argument(
                "--arch", choices=ARCH_MODEL_INV_REGISTRY[model_name]
            )
            group_args = parser.add_argument_group("Additional command-line arguments")
            MODEL_REGISTRY[model_name].add_args(group_args)
            globals()[model_name + "_parser"] = parser
