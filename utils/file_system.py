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



def savetodir(directory, data, name):
    ''' Saving data to a directory with specific filename

       :param directory: the directory to save the data to
       :param data: the list of data entries to save
       :param name: the filename
    '''
    with open(os.path.join(directory, name), 'w') as out:
        out.write('\n'.join([str(entry) for entry in data]))


def rmfile(path):
    if os.path.isfile(path):
        os.remove(path)

