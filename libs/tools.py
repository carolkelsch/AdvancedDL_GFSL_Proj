
from os.path import join, expanduser


def expanded_join(path, *paths):
    """Path concatenation utility function.
    Automatically handle slash and backslash depending on the OS system but also relative path user.

    :param path: Most left path.
    :param paths: Most right parts of path to concatenate.
    :return: A string which contains the absolute path.
    """
    return expanduser(join(path, *paths))