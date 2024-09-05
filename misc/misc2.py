#!/usr/bin/env python3
"""
Some wrapping around the default logging module.
"""

import logging
import sys
# from haggis.logs import add_logging_level
from termcolor import colored


class MyFormatter(logging.Formatter):
    """A nice formatter for logging messages."""
    line_formatting = f" {colored('(%(pathname)s:%(lineno)d)', 'light_grey')}"
    timestamp_formatting = f"{colored('[%(asctime)s]: ', 'green')}"
    trace_format = f"{colored('TRACE', 'cyan')}:{line_formatting} %(msg)s"
    debug_format = f"{colored('DEBUG', 'magenta')}:{line_formatting} %(msg)s"
    info_format = f"{colored('INFO', 'blue')}:{line_formatting} %(msg)s"
    print_format = f"%(msg)s"
    warning_format = f"{timestamp_formatting}{colored('WARNING', 'yellow')}:{line_formatting} %(msg)s"
    error_format = f"{timestamp_formatting}{colored('ERROR', 'red')}:{line_formatting} %(msg)s"
    critical_format = f"{timestamp_formatting}{colored('CRITICAL', 'red', attrs=['reverse', 'blink', 'bold'])}: {line_formatting} %(msg)s"

    def __init__(self):
        super().__init__(fmt=f"UNKNOWN: %(msg)s", datefmt=None, style='%')

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.TRACE:
            self._style._fmt = MyFormatter.trace_format
        elif record.levelno == logging.DEBUG:
            self._style._fmt = MyFormatter.debug_format
        elif record.levelno == logging.INFO:
            self._style._fmt = MyFormatter.info_format
        elif record.levelno == logging.PRINT:
            self._style._fmt = MyFormatter.print_format
        elif record.levelno == logging.WARNING:
            self._style._fmt = MyFormatter.warning_format
        elif record.levelno == logging.ERROR:
            self._style._fmt = MyFormatter.error_format
        elif record.levelno == logging.CRITICAL:
            self._style._fmt = MyFormatter.critical_format
        else:
            raise NotImplementedError(f"We don't know how to format logging levels: {record.levelno}")

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result


class StdOutFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno <= logging.PRINT


class StdErrFilter(logging.Filter):
    def filter(self, rec):
        stdout_filter = StdOutFilter()
        return not stdout_filter.filter(rec)


def setup_console_output():
    fmt = MyFormatter()
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    stdout_handler.addFilter(StdOutFilter())
    logging.root.addHandler(stdout_handler)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    stderr_handler.addFilter(StdErrFilter())
    logging.root.addHandler(stderr_handler)


def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs, stacklevel=2)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


addLoggingLevel('TRACE', logging.DEBUG - 5)
addLoggingLevel('PRINT', logging.WARNING - 5)

if __name__ == "__main__":
    logging.root.setLevel(logging.TRACE)
    setup_console_output()
    print("writing some logs...")
    logging.debug("something debug message")
    logging.trace("something trace message")
    logging.warning("something warning message")
