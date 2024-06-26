import os
import datetime


class SaveExp(object):
    """
    A class for saving experimental results in a specific directory
    according to common nomenclature
    """
    # author: Etienne Chollet

    def __init__(self, root_path):
        """
        Initialize object with a root directory path.

        Parameters
        ----------
        root_path : str
            the path to the directory where experiments will be saved.
        """
        self.root_path = root_path

    def nextDir(self):
        """
        Creates a new experiment folder in self.root_path according to
        naming system

        Returns
        -------
        exp_abs_path : str
            the absolute path to the new experiment folder.
        exp_dir_name : str
            the name of the new experiment folder.
        """
        # get list of all directories in root_path
        if os.path.isdir(self.root_path):
            preexisting_dirs = os.listdir(self.root_path)
        else:
            preexisting_dirs = []

        # start numbering experiments at 1
        exp_n = 1
        # name the first experiment directory "exp0001"
        exp_dir_name = "exp0001"

        # iterate until the next experiment number is found
        while exp_dir_name in preexisting_dirs:
            exp_n += 1
            exp_dir_name = "exp{:04d}".format(exp_n)

        # create the new experiment directory and get its absolute path
        exp_abs_path = f"{self.root_path}/{exp_dir_name}"
        os.makedirs(exp_abs_path, exist_ok=True)

        # print message indicating where experiment was saved
        print('\n', '#' * 50, f'\n\n SAVING EXPERIMENT TO: {exp_abs_path}\n\n',
              '#' * 50, '\n')

        return exp_abs_path, exp_dir_name

    def expDocumentation(self):
        """
        Creates a documentation file for experiment in experiment folder

        Returns
        -------
        exp_abs_path : str
            the absolute path to the new experiment folder.
        """
        # get the path and name of the new experiment directory
        exp_abs_path, exp_dir_name = self.nextDir()
        # create the path and name of the documentation file
        exp_documentation_abs_path = \
            f"{exp_abs_path}/{exp_dir_name}_documentation.txt"
        date = datetime.date.today()

        # create the documentation file and write the initial content
        file_obj = open(exp_documentation_abs_path, "x")
        file_obj.write(
            f"# Experiment Title: \n\n"
            f"# Date: {date}\n\n"
            f"# Objective:\n"
            f"- \n\n"
            f"# Notes:\n"
            f"-  ")
        file_obj.close()

        print(f'Edit experiment documentation at: '
              f'{exp_documentation_abs_path}\n')

        return exp_abs_path

    def main(self):
        """
        Calls the expDocumentation method to create the experiment
        directory and documentation file.

        Returns
        -------
        exp_abs_path : str
            the absolute path to the new experiment folder.
        """
        return self.expDocumentation()
