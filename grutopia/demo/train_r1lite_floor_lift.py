"""Train a phase-conditioned imitation policy from verified physical lifts."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('demonstrations', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    from grutopia_extension.manipulation.lift_policy import train_policy

    report = train_policy(args.demonstrations, args.output, args.epochs, args.seed)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
