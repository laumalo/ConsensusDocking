import os
import pytest
import mdtraj as md
from consensus_docking.preprocessing import Aligner


class TestAligner:
    """
    It wraps all tests that involve the Aligner class
    """

    @pytest.mark.parametrize("ref_pdb, ref_chain, query_pdb, query_chain",
                             [('aligned_2_chain_2.pdb', 'AB',
                               'not_aligned_2_chain_1.pdb', 'AB'),
                              ('aligned_2_chain_2.pdb', 'AB',
                               'not_aligned_2_chain_1.pdb', 'BA'),
                              ('aligned_2_chain_1.pdb', 'AB',
                               'not_aligned_2_chain_2.pdb', 'XY'),
                              ('aligned_2_chain_2.pdb', 'C',
                               'not_aligned_2_chain_1.pdb', 'C')])
    def test_can_align_chains(self, ref_pdb, ref_chain, query_pdb, query_chain):
        """
        Test that the aligner can superpose chains
        """
        relative_path = f'data/align/'
        ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                relative_path, ref_pdb)
        query_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  relative_path, query_pdb)
        a = Aligner(ref_path, ref_chain)
        a.align(query_path, query_chain, remove=False)

        out_file = query_path.replace('.pdb', '_align.pdb')
        ref = md.load(ref_path)
        out = md.load(out_file)
        rmsd = md.rmsd(out, ref, atom_indices=a.atoms_to_align_ref)[0]
        assert round(rmsd, 2) == 0.
        os.remove(out_file)

    @pytest.mark.parametrize("ref_pdb, ref_chain, query_pdb, query_chain",
                             [('aligned_2_chain_2.pdb', 'AB',
                               'not_aligned_2_chain_1.pdb', 'C'),
                              ('aligned_2_chain_2.pdb', 'C',
                               'not_aligned_2_chain_1.pdb', 'A')])
    def test_fail_alignment(self, ref_pdb, ref_chain, query_pdb, query_chain):
        """
        Test that the aligner can not superpose chains and raises an error
        """
        relative_path = f'data/align/'
        ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                relative_path, ref_pdb)
        query_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  relative_path, query_pdb)
        a = Aligner(ref_path, ref_chain)
        with pytest.raises(Exception):
            a.align(query_path, query_chain, remove=False)

