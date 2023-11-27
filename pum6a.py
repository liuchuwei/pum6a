# -*- coding: utf-8 -*-
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from modules import preprocess, train, predict, evaluate

modules = ['preprocess', 'train', 'predict', 'evaluate']

__version__ = "1.0.0"

def main():
    parser = ArgumentParser(prog='pum6a',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))

    subparsers = parser.add_subparsers(
        title='subcommands', description='valid commands',
        help='additional help', dest='command'
    )

    subparsers.required = True

    for module in modules:
        mod = globals()[module]
        p = subparsers.add_parser(module, parents=[mod.argparser()])
        p.set_defaults(func=mod.main)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()