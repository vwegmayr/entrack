"""Setup script for ml-project"""
import sys
import subprocess
from os.path import normpath
import argparse


def setup(args):
    """Setup function

    Requires:
        Installation of `miniconda`_.

    Todo:
        * Include automatic installtion of miniconda.

    .. _miniconda:
       https://conda.io/docs/install/quick.html#linux-miniconda-install

    """

    PROJECT_NAME = "entrack"

    if sys.version_info.major < 3:
        action = getattr(subprocess, "call")
    elif sys.version_info.minor < 5:
        action = getattr(subprocess, "call")
    else:
        action = getattr(subprocess, "run")

    if args.conda or args.all:
        action(["conda", "env", "create", "-n", PROJECT_NAME, "-f",
                ".environment"])

    if args.smt or args.all:
        action(["bash", "-c", "source activate {} && ".format(PROJECT_NAME) +
                "smt init -d {datapath} -i {datapath} -e python -m run.py "
                "-c error -l cmdline {project_name}".format(
                datapath=normpath('./data'), project_name=PROJECT_NAME)])

    if args.config or args.all:
        action(["cp", "configs/example_config.yaml", "configs/config.yaml"])

    if args.tractometer:
        if "tm_data_dir" not in args:
            raise ValueError("When installing the tractometer is mandatory to \
                              specify the installation directory.")
        _setup_tracto(action, tm_dir=args.tm_data_dir)


def _setup_tracto(action, tm_dir):
    """Run the setup the tractometer environment.

    The script downloads, installs and performs all the required setup for the
    tractometer tool.
    """
    # create environment
    action(["conda", "create", "-n", "entrack_tm", "python=2.7"])
    TM_DIR = normpath(tm_dir)
    tracto_install_script = "mkdir -p {} && ".format(TM_DIR) \
        + "cd {} && ".format(TM_DIR) \
        + "git clone https://github.com/scilus/ismrm_2015_tractography_challenge_scoring.git && " \
        + "pip install -r ismrm_2015_tractography_challenge_scoring/requirements.txt && " \
        + "pip install -r ismrm_2015_tractography_challenge_scoring/requirements_additional.txt" \

    tracto_download_script = "python setup.py build_all && " \
        + "wget www.tractometer.org/downloads/downloads/scoring_data_tractography_challenge.tar.gz && " \
        + "tar -xvf scoring_data_tractography_challenge.tar.gz && " \
        + "mv ./scripts/score_tractogram.py ./ && " \
        + "cd ../.." \

    # activate env and run the installation perations
    action([
        "bash", "-c",
        "source activate entrack_tm && {} &&  cd ismrm_2015_tractography_challenge_scoring && {}".
        format(tracto_install_script, tracto_download_script)
    ])
    print()
    print("=" * 80)
    print("Installation complete.")
    print("=" * 80)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Setup of conda and sumatra.")

    parser.add_argument("--smt", action="store_true")
    parser.add_argument("--conda", action="store_true")
    parser.add_argument("--config", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tractometer", action="store_true",
                        help="Setup tractometer environment inside the project. \
                              IMPORTANT Notice: The tractometer scoring tool \
                              requires a Python 2.7 environment to run. \
                              Therefore, a conda environment called \
                              'entrack_tm' will be created. This option is \
                              NOT installed with --all.")
    parser.add_argument("--tm_data_dir", type=str,
                        help="Specify the directory in which to unpack the TM \
                              data. This parameter is mandatory when running \
                              '--tractometer'.")
    args = parser.parse_args()

    setup(args)
