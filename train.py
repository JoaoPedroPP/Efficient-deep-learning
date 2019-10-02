import tensorflow as tf
import os
import models.net_factory as nf
import numpy as np
from keras import backend as K
from data_handler import DataHandler

flags = tf.app.flags

flags = DEFINE_integer('batchSize', 128, 'Batch Size')
flags = DEFINE_integer('numIter', 40000, 'Total de interações para treinamento')
flags = DEFINE_string('modelDir', 'model', 'Diretorio da rede')
flags = DEFINE_string('dataVersion', 'kitti2015', 'kitti2012 ou kitti2015')
flags = DEFINE_string('dataRoot', '/root/JP/training', 'Diretorio de treinamento')
flags = DEFINE_string('utilRoot', '/root/JP/cvpr16_stereo_public/preprocess/debug_15', 'Diretorio dos binarios')
flags = DEFINE_string('netType', 'win19_dep9', 'Tipo: win19_dep9')


flags = DEFINE_integer('evalSize', 200, 'Numero de validações por patch')
flags = DEFINE_integer('numTrImg', 160, 'Numero de imgens de treino')
flags = DEFINE_integer('numValImg', 40, 'Numero de imagens de teste')
flags = DEFINE_string('netType', 'win19_dep9', 'Tipo: win19_dep9')
flags = DEFINE_string('netType', 'win19_dep9', 'Tipo: win19_dep9')
flags = DEFINE_string('netType', 'win19_dep9', 'Tipo: win19_dep9')
