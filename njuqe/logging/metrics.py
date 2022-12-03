# Copyright (c) 2020 the NJUQE authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
A standalone module for aggregating metrics.

Metrics can be logged from anywhere using the `log_*` functions defined
in this module. The logged values will be aggregated dynamically based
on the aggregation context in which the logging occurs. See the
:func:`aggregate` context manager for more details.
"""

from .meters import LabelMeter
from fairseq.logging.metrics import get_active_aggregators


def log_labels(
    key: str,
    labels: list,
    priority: int = 10,
):
    """Log a scalar value.

    Args:
        key (str): name of the field to log
        labels (list): labels to save for f1 or others
        priority (int): smaller values are logged earlier in the output
    """
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, LabelMeter(), priority)
        agg[key].update(labels)
