import logging

class GlobalSettings(object):
    """Used to store global project settings."""
    def __init__(self):
        #Logger
        log_format = "[%(asctime)s.%(msecs)03d %(module)20s %(funcName)20s() %(levelname)8s] %(message)s"
        datefmt = '%Y-%m-%d %H:%M:%S'
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(format=log_format,
                            datefmt=datefmt)
        self.logger.setLevel(logging.DEBUG)

        # Bert tokenizer parameters

        self.bert_vocab_size = 8000
        self.bert_reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
        self.bert_tokenizer_params = dict(lower_case=True)

        # Hyperparameters
        self.MAX_TOKENS = 128
        self.BUFFER_SIZE = 20000
        self.BATCH_SIZE = 16

        self.NUM_EPOCHS = 300
        self.TRANSFORMER_NUM_LAYERS = 4 #4
        self.TRANSFORMER_D_MODEL = 256 # 128
        self.TRANSFORMER_DFF = 1024 # 512
        self.TRANSFORMER_NUM_HEADS = 8 # 8
        self.TRANSFORMER_DROPOUT_RATE = 0.1 # 0.1
