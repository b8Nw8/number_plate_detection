import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.models import Model, load_model
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.optimizers import Adam
from generate_data import TextImageGenerator

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def train(img_w, load=False):
    # Input Parameters
    img_h = 64

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    if keras.backend.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    batch_size = 32
    downsample_factor = pool_size ** 2
    tiger_train = TextImageGenerator('./train/', img_w, img_h, batch_size, downsample_factor)
    tiger_train.build_data()
    tiger_test = TextImageGenerator('./test/', img_w, img_h, batch_size, downsample_factor)
    tiger_test.build_data()

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    gru_1 = LSTM(rnn_size, return_sequences=True, name='gru1')
    gru_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, name='gru1_b')
    gru1_merged = Bidirectional(gru_1, backward_layer=gru_1b)(inner)
    gru_2 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')
    gru_2b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')
    gru2_merged = Bidirectional(gru_2, backward_layer=gru_2b)(gru1_merged)
    # transforms RNN output to character activations:
    inner = Dense(tiger_train.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(gru2_merged)
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[tiger_train.max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    sgd = Adam()
    if load:
        model = load_model('./tmp_model.h5', compile=False)
    else:
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    model.summary()
    if not load:
        # captures output of softmax so we can decode the output during visualization
        #test_func = tf.keras.backend.function([input_data], [y_pred])

        model.fit_generator(generator=tiger_train.next_batch(),
                        steps_per_epoch=tiger_train.n,
                        epochs=1,
                        validation_data=tiger_test.next_batch(),
                        validation_steps=tiger_test.n)

    return model


model = train(128, load=False)
extracted_layers = model.layers[:-4]
new_model = keras.Sequential(extracted_layers)
new_model.save('ocr_model.h5', include_optimizer=False)
