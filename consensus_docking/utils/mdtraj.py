import mdtraj as md 

def get_chains_dict(pdb):
        """
        It generates a dictionary with the chain indices (in Mdtraj) to chain
        names in the PDB file. 

        Parameters
        ----------
        pdb : str
            PDB file.

        Returns
        -------
        d : dict
            Dictionary chain idex/ID. 
        """
        from more_itertools import unique_everseen

        chain_indices = [chain.index for chain in md.load(pdb).topology.chains]
        with open(pdb, 'r') as f: 
            chains_pdb_ids = \
                list(unique_everseen([line[21:22] for line in f.readlines() 
                     if line.startswith('ATOM') or line.startswith('HETATOM')]))
        return {id:index for id,index in zip(chains_pdb_ids, chain_indices)} 