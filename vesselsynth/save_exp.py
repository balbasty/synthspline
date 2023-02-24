import os
import datetime

def nextExpDir():
    root_path = "/autofs/cluster/octdata3/users/epc28/data"
    preexisting_folders = os.listdir(root_path)
    i = 1
    name = "exp1"
    #date = datetime.date.today()

    while name in preexisting_folders:
        i += 1
        name = f"exp{i}"

    abs_path = f"{root_path}/{name}"
    os.mkdir(abs_path)
    return abs_path

def expDocumentation():
    exp_dir = nextExpDir()
    exp_documentation_abs_path = f"{exp_dir}/experiment_documentation.md"
    file_obj = open(exp_documentation_abs_path, "x")

    date = datetime.date.today()
    file_obj.write(f"# Experiment Title: \n\n# Date: {date}\n\n# Objective:\n- \n\n# Notes:\n-  ")
    file_obj.close()

    print('\n','#' * 30, f'\n\n SAVING TO: {exp_dir}\n\n','#' * 30,'\n')

    return exp_dir