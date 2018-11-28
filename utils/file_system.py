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

def save_pair_todir(directory, data, labels, data_outname, labels_outname):
    ''' Saving data and label pairs to a directory with specific filename
        ASSUMPTION: data contains more than or equal to labels amount of entries

       :param directory: the directory to save the data to
       :param data: the list of data entries to save
       :param labels: the list of label entries to save
       :param data_outname: the filename to save the data to
       :param labels_outname: the filename to save the labels to
    '''
    with open(os.path.join(directory, data_outname), 'w') as outD, open(os.path.join(directory, labels_outname), 'w') as outL:
        for d, l in zip(data, labels):
            for line in d.split('\n'):
                outD.write(line.lstrip().rstrip() + '\n')
                outL.write(str(l).lstrip().rstrip() + '\n')
