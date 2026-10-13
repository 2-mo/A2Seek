# Copyright (c) Alibaba, Inc. and its affiliates.
"""Command-line entry point for A2Seek inference."""

import _bootstrap  # Make the source package available from a checkout.
from a2seek.inference import main


if __name__ == "__main__":
    main()
