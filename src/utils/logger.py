import logging

def setup_log(log_file):
    logging.basicConfig(filename=log_file,level=logging.INFO,
                        format='%(asctime)s:(%levelname)s:%(message)s')
    return logging.getLogger('log')

logger=setup_log("../logs/project.log")
logger.info("start preprocess")
