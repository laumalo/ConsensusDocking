# Consensus docking



1. El script preprocessing/align_structures.py esta preparat per LightDock pero es pot generalitzar per tots. 

2. El script preprocessing/parse_scorings.py esta llest per ftdock, zdock, lightdock, frodock, patchdock, piper, 
   rosetta. Un exemple de command per corre el parser és:
   
   - ```python parse_scorings.py ftdock,zdock,lightdock,piper absp.ene,absp.ene,gso_100.out,ft.000.00 -w ../../```
   - ```python parse_scorings.py ftdock absp.ene -w ../../```
   
3. El script clustering/cluster_encoding_file.py esta llest per DBSCAN, OPTICS KMeans, donat un fitxer d'encoding. 
   Alguns exemples de com utilizar-lo (cal posar la llibreria pyyaml a l'env!!):
    
   - KMeans seleccionant columnes x1,x2 amb 2 clusters fent com a molt 400 iteracions i buscant 15 centroides (-n-init) 
     diferents, utilizant el valor default de output_dir ('./') per guardar yamls (ids de les poses a cada cluster 
     (defecte) + els ids de les poses del centroide (-scp)):
      
      ```python cluster_encoding_file.py python cluster_encoding_file.py kmeans ../../encodings/test.csv -sel-col x1,x2```
      ``` -nc 2 -max-iter 400 -n-init 15 -scp```
   - DBSCAN seleccionant coords de l'encoding, posant weights (tants com poses a l'encoding), amb epsilon 10, metric
     euclidean (default), guardant els ids de les poses de cada cluster (default) i els index (-scli). L'output dir es 
     la carpeta previa (../) des d'on s'executa:
    
     ```python cluster_encoding_file.py dbscan ../../encodings/test.csv -c -eps 10 -scli -o ../ -dw 1,1,1,1,1,1,2,1,2```
`
    
   - OPTICS seleccionant coords i sc, amb xi 0.2, metric minkowski amb extra parameter 1, poblacio minima de 2, 
     utilizant 2 jobs
    
     ```python cluster_encoding_file.py optics ../../encodings/test.csv -m minkowski -mp 1 -xi 0.2 -min-sam 2 -cs -nj 2```
    
> Nota: Per carregar els yaml files com a diccionaris a python cal correr:
>```
>import yaml
>with open('clust_poses_dict.yaml') as f:
>    d = yaml.full_load(f)
>```