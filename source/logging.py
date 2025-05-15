import logging.config
import os
from datetime import datetime

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '%(asctime)s [%(levelname)s] %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': os.path.join(LOG_DIR, f'trading_ai_{datetime.now().strftime("%Y%m%d")}.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': os.path.join(LOG_DIR, f'trading_ai_errors_{datetime.now().strftime("%Y%m%d")}.log'),
            'maxBytes': 10485760,
            'backupCount': 5
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file', 'error_file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger(__name__)