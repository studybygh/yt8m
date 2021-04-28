import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from custom_attention import PreUtil, MultiHeadAttention, EncoderLayer, DecoderLayer, Encoder, Decoder, Transformer

BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_LENGTH = 40

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(lang1.numpy()) + [tokenizer_pt.vocab_size+1]
    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang2.numpy()) + [tokenizer_en.vocab_size+1]
    return lang1, lang2

def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])
    return result_pt, result_en

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

# 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地
# 执行。该函数专用于参数张量的精确形状。为了避免由于可变序列长度或可变
# 批次大小（最后一批次较小）导致的再追踪，使用 input_signature 指定
# 更多的通用形状。
@tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64), tf.TensorSpec(shape=(None, None), dtype=tf.int64),])
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = PreUtil.create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,True,enc_padding_mask,combined_mask, dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)

def evaluate(inp_sentence):
    start_token = [tokenizer_pt.vocab_size]
    end_token = [tokenizer_pt.vocab_size + 1]

    # 输入语句是葡萄牙语，增加开始和结束标记
    inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # 因为目标是英语，输入 transformer 的第一个词应该是
    # 英语的开始标记。
    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = PreUtil().create_masks(encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,output,False,enc_padding_mask,combined_mask,dec_padding_mask)

        # 从 seq_len 维度选择最后一个词
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 如果 predicted_id 等于结束标记，就返回结果
        if predicted_id == tokenizer_en.vocab_size+1:
            return tf.squeeze(output, axis=0), attention_weights

        # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))
    sentence = tokenizer_pt.encode(sentence)
    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)
        # 画出注意力权重
        ax.matshow(attention[head][:-1, :], cmap='viridis')
        fontdict = {'fontsize': 10}
        ax.set_xticks(range(len(sentence)+2))
        ax.set_yticks(range(len(result)))
        ax.set_ylim(len(result)-1.5, -0.5)
        ax.set_xticklabels(['<start>']+[tokenizer_pt.decode([i]) for i in sentence]+['<end>'],fontdict=fontdict, rotation=90)
        ax.set_yticklabels([tokenizer_en.decode([i]) for i in result if i < tokenizer_en.vocab_size], fontdict=fontdict)
        ax.set_xlabel('Head {}'.format(head+1))
    plt.tight_layout()
    plt.show()

def translate(sentence, plot=''):
    result, attention_weights = evaluate(sentence)
    predicted_sentence = tokenizer_en.decode([i for i in result if i < tokenizer_en.vocab_size])
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))
    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)


if __name__ == "__main__":
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)
    tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)

    sample_string = 'Transformer is awesome.'
    tokenized_string = tokenizer_en.encode(sample_string)
    print ('Tokenized string is {}'.format(tokenized_string))

    original_string = tokenizer_en.decode(tokenized_string)
    print ('The original string: {}'.format(original_string))

    assert original_string == sample_string
    for ts in tokenized_string:
        print ('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
    # 将数据集缓存到内存中以加快读取速度。
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_examples.map(tf_encode)
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)
    pt_batch, en_batch = next(iter(val_dataset))
    print(pt_batch, en_batch)


    pos_encoding = PreUtil().positional_encoding(50, 512)
    print (pos_encoding.shape)
    plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()


    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    print(PreUtil().create_padding_mask(x))
    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    print(PreUtil().create_padding_mask(x))
    x = tf.random.uniform((1, 3))
    temp = PreUtil().create_look_ahead_mask(x.shape[1])
    print(temp)


    np.set_printoptions(suppress=True)
    temp_k = tf.constant([[10,0,0],
                          [0,10,0],
                          [0,0,10],
                          [0,0,10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[   1,0],
                          [  10,0],
                          [ 100,5],
                          [1000,6]], dtype=tf.float32)  # (4, 2)
    # 这条 `请求（query）符合第二个`主键（key）`，因此返回了第二个`数值（value）`。
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    PreUtil().print_out(temp_q, temp_k, temp_v)

    # 这条请求符合重复出现的主键（第三第四个），因此，对所有的相关数值取了平均。
    temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
    PreUtil().print_out(temp_q, temp_k, temp_v)

    # 这条请求符合第一和第二条主键，因此，对它们的数值去了平均。
    temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
    PreUtil().print_out(temp_q, temp_k, temp_v)

    temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)  # (3, 3)
    PreUtil().print_out(temp_q, temp_k, temp_v)


    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print(out.shape)
    print(attn.shape)


    sample_ffn = PreUtil().point_wise_feed_forward_network(512, 2048)
    print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)


    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)
    print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
    sample_decoder_layer = DecoderLayer(512, 8, 2048)
    sample_decoder_layer_output, _, _ = sample_decoder_layer(tf.random.uniform((64, 50, 512)), sample_encoder_layer_output, False, None, None)
    print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)


    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500, maximum_position_encoding=10000)
    sample_encoder_output = sample_encoder(tf.random.uniform((64, 62)), training=False, mask=None)
    print (sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8, dff=2048, target_vocab_size=8000, maximum_position_encoding=5000)
    output, attn = sample_decoder(tf.random.uniform((64, 26)), enc_output=sample_encoder_output, training=False, look_ahead_mask=None, padding_mask=None)
    print(output.shape, attn['decoder_layer2_block2'].shape)


    sample_transformer = Transformer(num_layers=2, d_model=512, num_heads=8, dff=2048,input_vocab_size=8500, target_vocab_size=8000,pe_input=10000, pe_target=6000)
    temp_input = tf.random.uniform((64, 62))
    temp_target = tf.random.uniform((64, 26))
    fn_out, _ = sample_transformer(temp_input, temp_target, training=False,enc_padding_mask=None,look_ahead_mask=None,dec_padding_mask=None)
    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)


    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8

    input_vocab_size = tokenizer_pt.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2
    dropout_rate = 0.1

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,epsilon=1e-9)
    temp_learning_rate_schedule = CustomSchedule(d_model)
    plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
                pe_input=input_vocab_size, pe_target=target_vocab_size, rate=dropout_rate)
    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(transformer=transformer,optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
    EPOCHS = 2
    for epoch in range(EPOCHS):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            if batch % 50 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,train_loss.result(), train_accuracy.result()))
        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    translate("este é um problema que temos que resolver.")
    print ("Real translation: this is a problem we have to solve .")

    translate("os meus vizinhos ouviram sobre esta ideia.")
    print ("Real translation: and my neighboring homes heard about this idea .")

    translate("vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.")
    print ("Real translation: so i 'll just share with you some stories very quickly of some magical things that have happened .")

    translate("este é o primeiro livro que eu fiz.", plot='decoder_layer4_block2')
    print ("Real translation: this is the first book i've ever done.")











