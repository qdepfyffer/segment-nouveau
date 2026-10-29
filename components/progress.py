"""
Creates a prefix to prepend to tqdm's progress bars so things look neater. Not strictly necessary and can be factored out in the future if desired.
"""

import logging


def console_formatter():
    """Returns the formatter of the console log handler, so progress bars match the console's log lines"""
    for handler in logging.getLogger().handlers:
        if type(handler) is logging.StreamHandler and handler.formatter is not None:
            return handler.formatter
    return logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")  # Fallback outside of Hydra

def bar_desc(logger, text):
    """Returns text with the same prefix as Hydra's log lines to be used as a prefix to the tqdm progress bar lines"""
    record = logger.makeRecord(logger.name, logging.INFO, "", 0, text, None, None)
    return console_formatter().format(record)