import os

def make_directory_structure(tmp_path, program, sc_filename='dock.sc'):
    base_path = str(tmp_path)
    program_folder = os.path.join(base_path, program)
    dummy_score_file = os.path.join(base_path, program, sc_filename)
    os.mkdir(program_folder)
    open(dummy_score_file, 'a').close()
    return base_path, program_folder, dummy_score_file
