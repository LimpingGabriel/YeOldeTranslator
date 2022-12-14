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
