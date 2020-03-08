from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG
import datetime
import os


def setup_logger(modname=__name__):
    logger = getLogger(modname)
    logger.setLevel(DEBUG)
    format = '%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s'

    sh = StreamHandler()
    sh.setLevel(DEBUG)
    formatter = Formatter(format)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # fh = file handler
    os.makedirs('log', exist_ok=True)
    fh = FileHandler('log/{}.log'.format(datetime.date.today()))
    fh.setLevel(DEBUG)
    fh_formatter = Formatter(format)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    return logger