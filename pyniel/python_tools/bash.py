import subprocess
import os

def bash(command):
    p = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ)
