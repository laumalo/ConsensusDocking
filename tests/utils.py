import os


def make_directory_structure(tmp_path, program, sc_filename='dock.sc',
                             is_lightdock=False):
    base_path = str(tmp_path)
    if not is_lightdock:
        program_folder = os.path.join(base_path, program)
    else:
        program_folder = os.path.join(base_path, program, 'swarm_0')
    dummy_score_file = os.path.join(base_path, program, sc_filename)
    os.makedirs(program_folder)
    open(dummy_score_file, 'a').close()
    return base_path, program_folder, dummy_score_file
