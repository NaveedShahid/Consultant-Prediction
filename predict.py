import argparse
import time
import numpy as np
import pickle
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_hub as hub

tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3'

def build_classifier_model(classes):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=False, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1024, activation='relu', name='dense')(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(classes, activation='sigmoid', name='classifier')(net)
    return tf.keras.Model(text_input, net)

def main(args):
    print('Loading model....')
    classifier_model = build_classifier_model(130)

    classifier_model.load_weights('classifier_model.hdf5')
    label_encoder = pickle.load(open('label_encoder.pkl','rb'))

    text= input('Please enter the description of the Task: \n')
    print(text)
    start = time.time()
    pred = classifier_model.predict([text])[0]
    end = time.time()
    pred = np.array(tf.math.top_k(pred,k=args.num,sorted=True)).T
    scores = pred[:,0]
    classes = pred[:,1].astype('int32')
    map={}
    for s,c in zip(scores,classes):
        c = label_encoder.inverse_transform([c])
        s= str(round(s*100))+'%'
        map.update({c[0]:s})
    print('Task: ', text,'\nPredicted Conultants: ',map)
    print('Inference Time: ', end - start)

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Consultant prediction')
    parser.add_argument('--num', type=int, required=False,default=3,
                        help='Number of consultants to predict')
    args = parser.parse_args()
    main(args)