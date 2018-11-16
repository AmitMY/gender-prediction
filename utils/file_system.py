import os, errno
from os.path import join, isfile


def makedir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def rmdir(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def listdir(directory, full=True):
    files = [f for f in os.listdir(directory) if isfile(join(directory, f))]
    if full:
        files = [join(directory, f) for f in files]
    return files
