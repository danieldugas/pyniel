import subprocess
import os

def bash(command):
    # stdout = subprocess.PIPE
    p = subprocess.run(command, shell=True, stdout=None, stderr=None, env=os.environ)
    return p.returncode
