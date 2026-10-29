"""
Provides git commit tracking to aid in tracing training outcomes back to their source. Not much reason to touch this file.
"""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # components/ -> actual project folder

def git_info() -> dict:  # get git info (lol)
    """
    Gets some git info to aid in tracing training outcomes back to their source.
    :return: A dictionary containing the current commit hash and a boolean value representing whether there are uncommitted changes in the current git repo
    """

    try:
        # Run the git commands to get the information we're concerned with: current commit hash and whether there are uncommitted changes
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout

    # If the git commands fail to run for whatever reason (git not installed, run outside of a git repo)
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": status != ""}  # If status is empty there are no uncommitted changes
