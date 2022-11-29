import tensorflow as tf
from GlobalSettings import GlobalSettings


settings = GlobalSettings()
class Translator(tf.Module):
    """description of class"""

    def __init__(self, tokenizers, translation_model):
        self.tokenizers = tokenizers
        self.translation_model = translation_model

    def __call__(self, sentence, max_length=settings.MAX_TOKENS):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.src.tokenize(sentence).to_tensor()
        encoder_input = sentence

        start_end = self.tokenizers.tar.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.translation_model([encoder_input, output], training=False)

            predictions = predictions[:, -1:, :]

            predicted_id = tf.argmax(predictions, axis=-1)

            output_array = output_array.write(i+1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())

        text = self.tokenizers.tar.detokenize(output)[0]
        tokens = self.tokenizers.tar.lookup(output)[0]


        self.translation_model([encoder_input, output[:, :-1]], training=False)
        attention_weights = self.translation_model.decoder.last_attn_scores

        return text, tokens, attention_weights
