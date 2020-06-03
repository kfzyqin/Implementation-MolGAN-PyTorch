import datetime
import string
import random


def get_date_postfix():
    """Get a date based postfix for directory name.

    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix


def random_string(string_len=3):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_len))