# CyclicOlefin_ReactivityRatio_Project
This repository contains the code for calculating reactivity ratios and extracting molecular descriptor properties of cyclic olefin compounds. For the previously calculated library of 869 cyclic olefin compounds, refer to https://descriptor-libraries.molssi.org/cyclic-olefins/

The files, directories, and their respective contents are listed below: 

1. `869 Cyclic Olefin Library`: Contains the 869 "Combined_CyclicOlefin_DFT_Library_Final.xlsx spreadsheet and the link to the MOLSSI library.
   
2. `Collect Properties`: Contains "GetProperties_CyclicOlefins.ipynb" python notebook for collecting properties on new compounds of interest and a folder of "Sample log files" for practicing implementation. Also included is the "gpenv_312.yml" file for creating the appropriate environment. 
   
    -Atom numbering and property collection was automated to support high-throughput library construction and analysis.
   
   <img width="601" height="451" alt="atommapping" src="https://github.com/user-attachments/assets/0e802246-847a-4a44-9745-a4f668d456aa" />

4. `Model Search`: Contains the "Data.xlsx" file containing all experimental data. Also, the model searching python notebook "Regression_modeling.ipynb" and the corresponding functions "utils.py" are enclosed herein.
   
5. `README.md`: Contains the instructions and summary of this repository. See SI <XXX> for further details. 


## Contributors
-Samantha E. Sloane, PhD

## References
-https://github.com/SigmanGroup/GetProperties
-https://github.com/SigmanGroup/chemical_space_and_clustering
-https://github.com/SigmanGroup/BO_modeling
