# Jupyter notebooks

[Jupyter notebooks](https://jupyter.org/) demonstrating use of the package.

These are tested against their current results using [nbval](https://nbval.readthedocs.io), and incorporated into the documentation using [nbsphinx](https://nbsphinx.readthedocs.io/).

## SARS-CoV-2 RBD data
These data are used for plausible simulated escape data against the RBD.

Data for antibodies targeting four "epitopes" on the SARS-CoV-2 RBD using the classification scheme of [Barnes et al (2020)](https://www.nature.com/articles/s41586-020-2852-1):
 - *LY-CoV016*: a "class 1" epitope
 - *LY-CoV555*: a "class 2" epitope
 - *REGN10987*: a "class 3" epitope
 - *CR3022*: a "class 4" epitope

The file [RBD_mutation_escape_fractions.csv](RBD_mutation_escape_fractions.csv) contains the mutation-level escape fractions for each antibody measured using deep mutational scanning in the following papers, only including mutations for which measurements are available for all four antibodies:
  - *LY-CoV016* and *REGN10987*: [Starr et al (2021), Science](https://science.sciencemag.org/content/371/6531/850)
  - *LY-CoV555*: [Starr et al (2021), Cell Reports Medicine](https://doi.org/10.1016/j.xcrm.2021.100255)
  - *CR3022*: [Greaney et al (2021), Cell Host & Microbe](https://www.sciencedirect.com/science/article/pii/S1931312820306247), but re-analyzed with the same expression and ACE2-binding cutoffs in [Starr et al (2021), Science](https://science.sciencemag.org/content/371/6531/850).

The file [RBD_mut_escape_df.csv](RBD_mut_escape_df.csv) contains mutation escape values (the "beta" values for the ``polyclonal`` package) generated from the mutation-level escape fractions using the script [RBD_mut_escape_df.py](RBD_mut_escape_df.py).

The file [RBD_activity_wt_df.csv](RBD_activity_wt_df.csv) contains the activity values for each epitope used in the simulations.

The file [RBD_seq.fasta](RBD_seq.fasta) is the coding sequence of the RBD used in the Bloom lab deep mutational scanning (optimized for yeast display).

The directory also contains [6M0J.pdb](6M0J.pdb), which is just a downloaded version of [PDB 6m0j](https://www.rcsb.org/structure/6M0J), which has the RBD in complex with ACE2.

## SARS-CoV-2 spike data
These are real data from deep mutational scanning:

 - [Lib-2_2022-06-22_thaw-1_LyCoV-1404_1_prob_escape.csv](Lib-2_2022-06-22_thaw-1_LyCoV-1404_1_prob_escape.csv) and [BA.1_site_numbering_map.csv](BA.1_site_numbering_map.csv) are real data from Omicron BA.1 spike deep mutational scanning.

 - [Omicron_BA.1_muteffects_observed.csv](Omicron_BA.1_muteffects_observed.csv) is functional effects of mutations to Omicron BA.1 spike from real data.
