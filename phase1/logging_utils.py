"""
日志配置工具
"""
import logging

def setup_logger(log_file='preprocess_debug.log'):
    """
    初始化日志记录器
    :param log_file: 输出日志文件路径
    :return: logger 实例
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
