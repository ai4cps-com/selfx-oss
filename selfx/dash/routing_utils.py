# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Nemanja Hranisavljevic
# Contact: nemanja@ai4cps.com


from urllib.parse import unquote
import datetime
from selfx.backend.datetime_utils import str_to_datetime


def parse_url(pathname):  # , href):
    feature = unquote(pathname[1:])
    split_url = feature.split('/')
    system = split_url[1]
    user = split_url[2]
    feature = split_url[3]
    start = split_url[4]
    end = split_url[5]
    return system, user, feature, start, end


def check_date(date_string):
    try:
        str_to_datetime(date_string)
    except:
        return False
    return True


def construct_url(system, user, feature, start, end):
    url = f'/dashboard/{system}/{user}/{feature}/{start}/{end}'
    return url


def construct_id(*args):
    return '-'.join(args).replace('_', '-').replace(' ', '-').replace('.', '-')


def get_today():
    return datetime.datetime.today().strftime('%Y-%m-%d')
