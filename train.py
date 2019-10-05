import tensorflow as tf
import os
import models.net_factory as nf
import numpy as np
from keras import backend as K
from data_handler import DataHandler

flags = tf.app.flags

flags.DEFINE_integer('batchSize', 128, 'Batch Size')
flags.DEFINE_integer('numIter', 40000, 'Total de interacoes para treinamento')
flags.DEFINE_string('modelDir', 'model', 'Diretorio da rede')
flags.DEFINE_string('dataVersion', 'kitti2015', 'kitti2012 ou kitti2015')
flags.DEFINE_string('dataRoot', '/root/JP/training', 'Diretorio de treinamento')
flags.DEFINE_string('utilRoot', '/root/JP/cvpr16_stereo_public/preprocess/debug_15', 'Diretorio dos binarios')
flags.DEFINE_string('netType', 'win19_dep9', 'Tipo: win19_dep9')


flags.DEFINE_integer('evalSize', 200, 'Numero de validacoes por patch')
flags.DEFINE_integer('numTrImg', 160, 'Numero de imgens de treino')
flags.DEFINE_integer('numValImg', 40, 'Numero de imagens de teste')
flags.DEFINE_integer('patchSize', 37, 'Tamanho do Patch de treinamento')
flags.DEFINE_integer('numValLoc', 50000, 'Numero de validacoes (locais)')
flags.DEFINE_integer('dispRange', 201, 'Range da disparidade')
flags.DEFINE_string('phase', 'train', 'Treino ou teste')

FLAGS = flags.FLAGS

np.random.seed(123)

dhandler = DataHandler(dataVersion=FLAGS.dataVersion,
        dataRoot=FLAGS.dataRoot,
        utilRoot=FLAGS.utilRoot,
        numTrImg=FLAGS.numTrImg,
        numValImg=FLAGS.numValImg,
        numValLoc=FLAGS.numValLoc,
        batchSize=FLAGS.batchSize,
        patchSize=FLAGS.patchSize,
        dispRange=FLAGS.dispRange)

if FLAGS.dataVersion == 'kitti2012':
    numChannels = 1
elif FLAGS.dataVersion == 'kitti2015':
    numChannels = 3
else:
    sys.exit('Versao invalida')

def train():
    if not os.path.exists(FLAGS.modelDir):
        os.makedirs(FLAGS.modelDir)

    g = tf.Graph()
    with g.as_default():
        lImage = tf.compat.v1.placeholder(tf.float32, [None, FLAGS.patchSize, FLAGS.patchSize, numChannels], name='lImage')
        rImage = tf.compat.v1.placeholder(tf.float32, [None, FLAGS.patchSize, FLAGS.patchSize+FLAGS.dispRange-1, numChannels], name='rImage')
        targets = tf.compat.v1.placeholder(tf.float32, [None, FLAGS.dispRange], name='targets')

        #snet = nf.Create(lImage, rImage, targets, FLAGS.netType, FLAGS.patchSize, FLAGS.dispRange, FLAGS.dataVersion)
        snet = nf.Create(lImage, rImage, targets, FLAGS.netType)

        loss = snet['loss']
        trainStep = snet['trainStep']
        session = tf.InteractiveSession()
        session.run(tf.global_varibles_initializer())
        saver = tf.train.Saver(max_to_keep=1)

        accLoss = tf.compat.v1.placeholder(tf.float32, shape=())
        lossSummary = tf.summary.scalar('loss', accLoss)
        trainWriter = tf.summary.FileWriter(FLAGS.modelDir+'/training', g)

        saver = tf.train.Saver(max_to_keep=1)
        losses = []
        summaryIndex = 1
        lRate = 1e-2

        for it in range(1, FLAGS.numIter):
            lPatch, rPatch, patchTargets = dhandler.NextBatch()

            trainDict = {lImage:lPatch, rImage:rPatch, targets:patchTargets, snet['lrate']:lRate, K.learning_phase():1}
            _, miniLoss = session.run([trainStep, loss], feed_dict=trainDict)
            losses.append(miniLoss)

            if it%100 == 0:
                print('Loss as step: %d: %.6f'%(it, miniLoss))
                saver.save(session, os.path.join(FLAGS.modelDir, 'model.ckpt'), global_step=snet['globalStep'])
                trainSummary = session.run(lossSummary, feed_dict={accLoss:np.mean(losses)})
                trainWriter.add_summary(trainSummary, summaryIndex)
                summaryIndex += 1
                trainWriter.flush()
                losses = []

                if it == 24000:
                    lRate = lRate/5
                elif it > 24000 and (it-24000)%8000 == 0:
                    lRate = lRate/5

def evaluate():
    lPatch, rPatch, patchTargets = dhandler.Evaluate()
    labels = np.argmax(patchTargets, axis=1)

    with tf.Session() as session:
        lImage = tf.compat.v1.placeholder(tf.float32, [None, FLAGS.patchSize, FLAGS.patchSize, numChannels], name='lImage')
        rImage = tf.compat.v1.placeholder(tf.float32, [None, FLAGS.patchSize, FLAGS.patchSize+FLAGS.dispRange-1, numChannels], name='rImage')
        targets = tf.compat.v1.placeholder(tf.float32, [None, FLAGS.dispRange], name='targets')

        snet = nf.create(lImage, rImage, targets, FLAGS.netType, FLAGS.patchSize, FLAGS.dispRange, FLAGS.dataVersion)
        prod = snet['innerProdut']
        predicted = tf.argmax(prod, axis=1)
        accCount = 0

        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(FLAGS.modelDir))

        for i in range(0, lPatch.shape[0], FLAGS.evalSize):
            evalDict = {lImage:lPatch[i:i+FLAGS.evalSize], rImage:rPatch[i:i+FLAGS.evalSize], K.learning_phase():0}
            pred = session.run([predicted], feed_dict=evalDict)
            accCount += np.sum(np.abs(pred-labels[i:i+FLAGS.evalSize])<=3)
            print('iter. %d finished, with %d correct (3-pixel error)'%(i+1, accCount))

        print('Accuracy: %.3f'%((accCount/lPatch.shape[0])*100))

if FLAGS.phase == 'train':
    train()
elif FLAGS.phase == 'evaluate':
    evaluate()
else:
    sys.exit('train or evaluate')

