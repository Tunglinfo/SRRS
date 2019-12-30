class Config(object):
    DEBUG = False
    TESTING = False

    # flask-cacheing
    CACHE_TYPE = 'redis'
    CACHE_DEFAULT_TIMEOUT = 10
    CACHE_REDIS_URL = 'redis://localhost:6379/0'


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

    ## Server

    pass

class DevelopmentConfig(Config):
    DEBUG = True

    ## Server

    pass

class TestingConfig(Config):
    TESTING = True

    pass

config = {
    'development': DevelopmentConfig,
    'test': TestingConfig,
    'production': ProductionConfig,
    # defult config
    'default': DevelopmentConfig
}