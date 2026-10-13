"""Command-line entry point for A2Seek training."""

import _bootstrap  # Make the source package available from a checkout.
from a2seek.training import build_command, main


if __name__ == "__main__":
    main()
