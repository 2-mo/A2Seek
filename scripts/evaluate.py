"""Command-line entry point for A2Seek evaluation."""

import _bootstrap  # Make the source package available from a checkout.
from a2seek.evaluation import evaluate, main


if __name__ == "__main__":
    main()
