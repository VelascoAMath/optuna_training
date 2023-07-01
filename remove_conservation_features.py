'''
This removes all of the conservation-related features from a dataset

@author: Alfredo Velasco
'''

import os
import pandas as pd

non_conservation_feature_list = [
	# 'source',
	'protein_id',
	'protein_position',
	'reference_aa',
	'mutant_aa',
	'label',
	# 'score',
	'concavity_track',
	'canbind_track',
	'domain_track',
	'interacdome_dna_track',
	'interacdome_rna_track',
	'interacdome_sm_track',
	'interacdome_ion_track',
	'interacdome_peptide_track',
	'dsprint_track',
	'phosphorylation_track',
	'disorder_short_track',
	'disorder_long_track',
	'disorder_structured_track',
	'ref_alanine',
	'ref_cysteine',
	'ref_aspartate',
	'ref_glutamate',
	'ref_phenylalanine',
	'ref_glycine',
	'ref_histidine',
	'ref_isoleucine',
	'ref_lysine',
	'ref_leucine',
	'ref_methionine',
	'ref_asparagine',
	'ref_proline',
	'ref_glutamine',
	'ref_arginine',
	'ref_serine',
	'ref_threonine',
	'ref_valine',
	'ref_tryptophan',
	'ref_tyrosine',
	'mut_alanine',
	'mut_cysteine',
	'mut_aspartate',
	'mut_glutamate',
	'mut_phenylalanine',
	'mut_glycine',
	'mut_histidine',
	'mut_isoleucine',
	'mut_lysine',
	'mut_leucine',
	'mut_methionine',
	'mut_asparagine',
	'mut_proline',
	'mut_glutamine',
	'mut_arginine',
	'mut_serine',
	'mut_threonine',
	'mut_valine',
	'mut_tryptophan',
	'mut_tyrosine',
	'ref_func_group',
	'mut_func_group',
	'ref_charge',
	'mut_charge',
	'ref_kyte_doolittle',
	'mut_kyte_doolittle',
	'ref_alpha_helix_prop',
	'mut_alpha_helix_prop',
	'ref_beta_sheet_prop',
	'mut_beta_sheet_prop',
	'ref_beta_turn_prop',
	'mut_beta_turn_prop',
	'ref_volume',
	'mut_volume',
	'ref_volume_group',
	'mut_volume_group',
	'ref_hbond_donor',
	'mut_hbond_donor',
	'ref_hbond_acceptor',
	'mut_hbond_acceptor',
	'Non-standard residue',
	'Active site',
	'Binding site',
	'DNA binding',
	'Metal binding',
	'Nucleotide binding',
	'Site',
	'Intramembrane',
	'Topological domain',
	'Transmembrane',
	'Chain',
	'Glycosylation',
	'Lipidation',
	'Modified residue',
	'Peptide',
	'Propeptide',
	'Signal peptide',
	'Transit peptide',
	'Compositional bias',
	'Domain [FT]',
	'Motif',
	'Region',
	'Repeat',
	'Zinc finger'
]


conservation_feature_list = [
	'conservation_track',
	'Natural variant',
	'demask_score',
	'demask_entropy',
	'demask_log2f_var',
	'demask_matrix'
]


def remove_conservation(dataset_f_name):
	'''
	Removes the conservation features from a specified dataset path

	:param str dataset_f_name: the path to the dataset
	:raises Excpeption: if dataset_f_name is not a tsv file
	'''
	dataset_name, dataset_ext = os.path.splitext(dataset_f_name)

	if dataset_ext.lower() != '.tsv':
		raise Exception("Inputted file name ({dataset_f_name}) must be a tsv file!")

	output_f_name = f"{dataset_name}_No_Con.tsv"


	df = pd.read_csv(dataset_f_name, sep='\t')
	print(f"{dataset_f_name} -> {output_f_name}")
	
	print(df)

	df_out = df[non_conservation_feature_list]

	print(df_out)
	print()
	df_out.to_csv(path_or_buf=output_f_name, sep='\t', index=False, header=True)




def main():
	dataset_list = [
	# 'datasets/DRGN_minus_mmc2_PhysChem_Intersect.tsv', 'datasets/DRGN_minus_mmc2_PhysChem.tsv',
	# 'datasets/DRGN_PhysChem_Intersect.tsv', 'datasets/DRGN_PhysChem.tsv',
	'datasets/mmc2_PhysChem_3_pos_neg.tsv', 'datasets/mmc2_PhysChem_Intersect.tsv', 'datasets/mmc2_PhysChem.tsv',
	# 'temp/mavedb_BERT_mut_PhysChem.tsv'
	'datasets/docm/docm_PhysChem_Intersect.tsv',
	'datasets/docm/docm_minus_mmc2_PhysChem.tsv',
	'datasets/docm/docm_minus_mavedb_PhysChem.tsv',
	]


	for dataset in dataset_list:
		remove_conservation(dataset)

if __name__ == '__main__':
	main()