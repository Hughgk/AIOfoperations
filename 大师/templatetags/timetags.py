from django import template
import time
register = template.Library()


def print_timestap(timestamp):
    try:
        # assume, that timestamp is given in seconds with decimal point
        ts = float(timestamp)
    except ValueError:
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts))

register.filter(print_timestap)