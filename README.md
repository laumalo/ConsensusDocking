# Consensus docking



1. El script preprocessing/align_structures.py esta preparat per LightDock pero es pot generalitzar per tots. 

2. El script preprocessing/parse_scorings.py esta llest per ftdock, zdock, lightdock, frodock, patchdock, piper, 
   rosetta. Un exemple de command per corre el parser és:
   
   - `python parse_scorings.py ftdock,zdock,lightdock,piper absp.ene,absp.ene,gso_100.out,ft.000.00 -w ../../`
   - `python parse_scorings.py ftdock absp.ene -w ../../`