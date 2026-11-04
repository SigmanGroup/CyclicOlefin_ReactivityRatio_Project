Property Collection Workflow
1. The SMARTS pattern ‘CC=C[#6,#16]’ was used to identify atom numbers and elements appearing in or adjacent to olefins. 
2. Skipped all compounds containing multiple olefins except for those that were evaluated experimentally, T22, E1, and E4. For those compounds, atom mapping was done manually to ensure properties were collected on the correct olefin.
3. Olefinic positions were distinguished electronically by comparing Boltzmann weighted average Hirshfeld atomic charges across all conformers, where C1 is designated as the position with the more negative partial charge, or more electron rich.
4. A series of electronic, steric, and geometric properties were computed and/or collected for each conformer using GetProperties18 and MORFEUS19 packages.
5. A quasi-harmonic thermochemical correction was applied to the energies associated with each conformer using GoodVibes20. Collected properties were weighted across conformational ensembles. The resulting feature sets included feature maximum, minimum, mean, range, standard deviations, as well as a features from the lowest energy conformer and an approximation to a Boltzmann-weighted average. 
6. Additionally, symmetry-related averages of allylic and olefinic positions were computed as averages, differences, minimums, and maximums between related positions. 
7. The descriptor set was curated to improve interpretability, including the removal of maximum, minimum, standard deviation, and range statistics for frontier orbital properties.
8. Monomer-level descriptors were combined to generate descriptors for each monomer pair represented in the training and test datasets.
9. The corresponding experimental data were added to the combined descriptor tables.
10. A collinearity cutoff of 0.85 was applied. Within each correlated group, the descriptor most strongly correlated with ln⁡(r) was retained, reducing the feature set from 1,624 to 136 descriptors.
