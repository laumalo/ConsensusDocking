import os
import pytest
import mdtraj as md
from consensus_docking.preprocessing import Aligner


@pytest.fixture
def relative_path():
    return f'data/align/'


class TestAligner:
    """
    It wraps all tests that involve the Aligner class
    """

    @pytest.mark.parametrize("ref_pdb, ref_chain, query_pdb, query_chain",
                             [('aligned_2_chain_2.pdb', ['A','B'],
                               'not_aligned_2_chain_1.pdb', ['A','B']),
                              ('aligned_2_chain_2.pdb', ['A','B'],
                               'not_aligned_2_chain_1.pdb', ['B','A']),
                              ('aligned_2_chain_1.pdb', ['A','B'],
                               'not_aligned_2_chain_2.pdb', ['X','Y']),
                              ('aligned_2_chain_2.pdb', ['C'],
                               'not_aligned_2_chain_1.pdb', ['C'])])
    def test_can_align_chains(self, relative_path, ref_pdb, ref_chain,
                              query_pdb, query_chain):
        """
        Test that the aligner can superpose chains
        """
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
                             [('aligned_2_chain_2.pdb', ['A','B'],
                               'not_aligned_2_chain_1.pdb', ['C']),
                              ('aligned_2_chain_2.pdb', ['C'],
                               'not_aligned_2_chain_1.pdb', ['A'])])
    def test_fail_alignment(self, relative_path, ref_pdb, ref_chain, query_pdb,
                            query_chain):
        """
        Test that the aligner can not superpose chains and raises an error
        """
        ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                relative_path, ref_pdb)
        query_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  relative_path, query_pdb)
        a = Aligner(ref_path, ref_chain)
        with pytest.raises(Exception):
            a.align(query_path, query_chain, remove=False)

         
    @pytest.mark.parametrize("ref_pdb, ref_chain, query_chain",
                             [('aligned_2_chain_2.pdb', ['C'], ['C'])])

    def test_run_alignment_in_folder(self, relative_path, ref_pdb, ref_chain,
                                     query_chain):

        import tempfile 

        ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                relative_path, ref_pdb)
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   relative_path)
        
        a = Aligner(ref_path, ref_chain)
        a.run_aligment(folder_path, query_chain, remove=False,
                       prefix_file='test_run_alignment')


        out_file_1 = os.path.join(folder_path, 'test_run_alignment_1_align.pdb')
        out_file_2 = os.path.join(folder_path, 'test_run_alignment_2_align.pdb')
        aligned_files = [out_file_1, out_file_2]
        
        ref = md.load(ref_path)
        rmsd_1 = md.rmsd(md.load(out_file_1), ref, 
                         atom_indices=a.atoms_to_align_ref)[0]
        rmsd_2 = md.rmsd(md.load(out_file_2), ref, 
                         atom_indices=a.atoms_to_align_ref)[0]
        print(rmsd_1, rmsd_2)
        assert round(rmsd_1, 2) == 0.
        assert round(rmsd_2, 2) == 0.
        #os.remove(out_file_1)
        #os.remove(out_file_2)

