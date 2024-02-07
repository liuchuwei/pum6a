# -*- coding: utf-8 -*-

# https://gitlab.com/piroonj/eligos2
#!/usr/bin/env python
"""
.. module:: main
   :platform: Unix, MacOSX
   :synopsis: Main entry of the script.

.. moduleauthor:: Piroon Jenjaroenpun <piroonj@gmail.com>

"""
import argparse
import sys
import os
from Eligos._version import ELIGOS_VERSION
from Eligos import _option_parsers

class SubcommandHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _format_action(self, action):
        parts = super(SubcommandHelpFormatter, self)._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = "\n".join(parts.split("\n")[1:])
        return parts

def main(args=None):
	if args is None:
	    args = sys.argv[1:]

	function_commands = [('map_preprocess','Preprocess mapped reads',
		 _option_parsers.get_map_preprocess_parser()),
		('build_genedb', 'Build bam files from gene database',
		 _option_parsers.get_build_genedb_parser()),
		('rna_mod', 'Identify RNA modification against rBEM5+2 model',
		 _option_parsers.get_rna_mod_parser()),
		('pair_diff_mod', 'Identify RNA modification against control condition',
		 _option_parsers.get_pair_diff_mod_parser()),
		('bedgraph', 'Filtering and creating BedGraph for IGV plot',
		 _option_parsers.get_bedgraph_parser()),
		('filter', 'Filtering for Eligos result ',
		 _option_parsers.get_filter_parser()),
		('multi_samples_test', 'Test multiple samples of rna_mod result ',
		 _option_parsers.get_multi_samples_test_parser()),
		]

	desc = ('ELIGOS command groups:\n' + '\n'.join([
	            '\t{0: <25}{1}'.format(grp_name, grp_help)
	            for grp_name, grp_help, _ in function_commands ]))

	parser = argparse.ArgumentParser(
	    prog='eligos',
	    description='ELIGOS is a package of tools ' +
	    'for the identification of modified nucleotides  \n' +
	    "based on nanopore sequencing data.\n\n" + desc,
		formatter_class=SubcommandHelpFormatter)
		# formatter_class=argparse.RawTextHelpFormatter)

	parser.add_argument(
	    '-v', '--version', action='version',
	    version='ELIGOS version: {}'.format(ELIGOS_VERSION),
	    help='show ELIGOS version and exit.')

	# Command groups
	service_subparsers = parser.add_subparsers(dest="service_command")

	# add preprocess command
	for grp_name, grp_help, grp_sub_cmds in function_commands:
		grp_parser = service_subparsers.add_parser(
			grp_name, parents=[grp_sub_cmds,],add_help=False)
		grp_parser.set_defaults(action_command=grp_name)

	# add preprocess in both the service parser and action parser
	# process_parser.set_defaults(action_command=function_commands[0][0])
	try:
		save_args = args
		args = parser.parse_args(args)
	except:
		raise

	if args.service_command is None:
		print('\n*********************************************')
		print('error: Must provide a ELIGOS command group.')
		print('\n*********************************************\n')
		save_args.append('-h')
		parser.parse_args(save_args)
		parser.print_help()

	if args.action_command is None:
	    save_args.append('-h')
	    parser.parse_args(save_args)

	if args.action_command == 'map_preprocess':
		import _map_preprocess
		if args.bam is None:
			print('\n***********************************')
			print('Eligos error: Must provide BAM file')
			print('\n***********************************\n')
			save_args.append('-h')
			parser.parse_args(save_args)
			parser.print_help()
		else:
			_map_preprocess._map_preprocess_main(args)

	elif args.action_command == 'build_genedb':
		import _build_genedb
		if len(args.bam) == 0 or args.region is None:
			print('\n*************************************************************')
			print('Eligos error: Must provide Regions (BED/GTF/GFF) and BAM file')
			print('\n*************************************************************\n')
			save_args.append('-h')
			parser.parse_args(save_args)
			parser.print_help()
		else:
			_build_genedb._build_genedb_main(args)

	elif args.action_command == 'rna_mod':
		from Eligos import _rna_mod
		if len(args.bam) == 0 or args.region is None or args.reference is None:
			print('\n*********************************************************************************')
			print('Eligos error: Must provide Regions (BED6/BED12), BAM and Reference sequence file')
			print('\n*********************************************************************************\n')
			save_args.append('-h')
			parser.parse_args(save_args)
			parser.print_help()
		else:
			_rna_mod._rna_mod_main(args)

	elif args.action_command == 'pair_diff_mod':
		import _pair_diff_mod
		if len(args.test_bams) == 0 or len(args.ctrl_bams) == 0  or args.region is None or args.reference is None:
			print('\n*********************************************************************************')
			print('Eligos error: Must provide Regions (BED6/BED12), test and control BAM and Reference sequence file')
			print('\n*********************************************************************************\n')
			save_args.append('-h')
			parser.parse_args(save_args)
			parser.print_help()
		else:
			_pair_diff_mod._pair_diff_mod_main(args)

	elif args.action_command == 'bedgraph':
		import _bedgraph
		if args.input is None:
			print('\n*************************************************************')
			print('Eligos error: Must provide the Eligos result')
			print('\n*************************************************************\n')
			save_args.append('-h')
			parser.parse_args(save_args)
			parser.print_help()
		else:
			_bedgraph._bedgraph_main(args)

	elif args.action_command == 'filter':
		import _filter
		if args.input is None:
			print('\n*************************************************************')
			print('Eligos error: Must provide the Eligos result')
			print('\n*************************************************************\n')
			save_args.append('-h')
			parser.parse_args(save_args)
			parser.print_help()
		else:
			_filter._filter_main(args)
	# print("Done!")

	elif args.action_command == 'multi_samples_test':
		import _multi_samples_test
		if len(args.test_mods) != len(args.ctrl_mods) or len(args.test_mods) <= 1:
			print('\n*********************************************************************************')
			print('Eligos error: Must provide multiple rna_mod results with ')
			print('the same number of samples between test and control.')
			print('\n*********************************************************************************\n')
			save_args.append('-h')
			parser.parse_args(save_args)
			parser.print_help()
		else:
			_multi_samples_test._multi_samples_test_main(args)

if __name__ == "__main__":
    main()
