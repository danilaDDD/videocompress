try:
    from tensorflow.keras import models as mds
    from tensorflow.keras import layers as lrs
    from tensorflow.keras import losses, optimizers, metrics
    print('use tensorflow backend')

except ImportError:

    try:
        import plaidml.keras
        plaidml.keras.install_backend()

        from keras import models as mdls
        from keras import layers as lrs
        from keras import losses, optimizers, metrics

        print('plaidml backend')

    except ImportError:
        raise ImportError('not import backend')


