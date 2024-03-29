import tensorflow as tf
import models.win19_dep9 as net19
from tensorflow.python.ops import control_flow_ops

from keras import backend as K
from keras.losses import categorical_crossentropy
#import keras
slim = tf.contrib.slim

def ThreePixelError(lbranch, rbranch, targets):
    print(rbranch.shape)
    l = tf.squeeze(lbranch)
    r = tf.transpose(tf.squeeze(rbranch), perm=[0,2,1])
    prod = tf.matmul(l, r) # Executa o produto escalar das imagens
    print(prod)
    prodFlatten = tf.contrib.layers.flatten(prod)
    #prodFlatten = tf.keras.layers.Flatten(prod)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=prodFlatten), name='loss')

    return prodFlatten, loss

def Create(limage, rimage, targets, netType, patchSize=19, dispRange=201, dataVersion='kitti2015'):
    if dataVersion == 'kitti2012':
        numChannels = 1
    elif dataVersion == 'kitti2015':
        numChannels = 3
    else:
        sys.exit('dataVersion deve ser \'kitti2012\' ou \'kitti2015\'')

    leftInputShape = (patchSize, patchSize, numChannels)
    rightInputShape = (patchSize, patchSize+dispRange-1, numChannels)

    with tf.name_scope('siamese_' + netType):
        if netType == 'win37_dep9':
            lbranch = net37.CreateNetwork(limage, leftInputShape)
            rbranch = net37.CreateNetwork(rimage, rightInputShape)
        elif netType == 'win29_dep9':
            lbranch = net29.CreateNetwork(limage, leftInputShape)
            rbranch = net29.CreateNetwork(rimage, rightInputShape)
        elif netType == 'win19_dep9':
            lbranch = net19.CreateNetwork(limage, leftInputShape)
            rbranch = net19.CreateNetwork(rimage, rightInputShape)
        elif netType == 'win9_dep9':
            lbranch = net9.CreateNetwork(limage, leftInputShape)
            rbranch = net9.CreateNetwork(rimage, rightInputShape)
        else:
            sys.exit('Modelo invalido')
#        print(targets.shape)
        prodFlatten, loss = ThreePixelError(lbranch, rbranch, targets)
        lrate = tf.compat.v1.placeholder(tf.float32, [], name='lrate')
        with tf.name_scope('optimizer'):
            globalStep = tf.get_variable('globalStep', [], initializer=tf.constant_initializer(0.0), trainable=False)
            optimizer=tf.train.AdagradOptimizer(lrate)
            trainStep = slim.learning.create_train_op(loss, optimizer, global_step=globalStep)
            updateOps = tf.get_collection(tf.GrapKeys.UPDATE_OPS)

            if updateOps:
                update = tf.group(*updateOps)
                loss = control_flow_ops.with_dependencies([update], loss)

        net = {'lbranch': lbranch, 'rbranch': rbranch, 'loss': loss, 'innerProduct': prodFlatten, 'trainStep': trainStep, 'globalStep': globalStep, 'lrate': lrate}
        return net

def MapInnerProduct(lmap, rmap):
    prod = tf.reduce_sum(tf.multiply(lmap, rmap), axis=3, name='map_inner_product')
    return prod
