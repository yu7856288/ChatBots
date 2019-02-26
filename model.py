import time

import numpy as np
import tensorflow as tf

import config

class ChatBotModel:
    def __init__(self, forward_only, batch_size):
        """forward_only: if set, we do not construct the backward pass in the model. 只有前向，或者 含有backward pass
        """
        print('Initialize new model')
        self.fw_only = forward_only ####只有前向传播
        self.batch_size = batch_size### batch size的大小

    def _create_placeholders(self):
        # Feeds for inputs. It's a list of placeholders
        print('Create placeholders')
        ###创建的placeholders  encoderinputs 是一个句子，长度是 bucket的长度
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                               for i in range(config.BUCKETS[-1][0])]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                               for i in range(config.BUCKETS[-1][1] + 1)] ### decoder inputs 加1， 多加的1是首尾符
        self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                              for i in range(config.BUCKETS[-1][1] + 1)] ### decoder masks 长度加1,

        # Our targets are decoder inputs shifted by one (to ignore <GO> symbol)
        self.targets = self.decoder_inputs[1:]###把decoder 左移一位，就是decoder ouput targe

    def _inference(self):####sample softmax 与实际的vocab size 作映射
        print('Create inference')
        # If we use sampled softmax, we need an output projection.
        # Sampled softmax only makes sense if we sample less than vocabulary size.####当sample个数小于vocab 时，做projection
        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < config.DEC_VOCAB:
            w = tf.get_variable('proj_w', [config.HIDDEN_SIZE, config.DEC_VOCAB])
            b = tf.get_variable('proj_b', [config.DEC_VOCAB])
            self.output_projection = (w, b)

        def sampled_loss(logits, labels):
            labels = tf.reshape(labels, [-1, 1]) ###做reshape
            return tf.nn.sampled_softmax_loss(weights=tf.transpose(w), ####tf.transpose(w) 做transpose？ shape的要求
                                              biases=b, 
                                              inputs=logits, 
                                              labels=labels, 
                                              num_sampled=config.NUM_SAMPLES,  ### num samples 是采样数
                                              num_classes=config.DEC_VOCAB)    ### dec vocab 是最终分类个数
        self.softmax_loss_function = sampled_loss

        single_cell = tf.contrib.rnn.GRUCell(config.HIDDEN_SIZE) ### 使用GRU
        self.cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(config.NUM_LAYERS)]) ###Rnn cell 堆叠 GRU cell

    def _create_loss(self):
        print('Creating loss... \nIt might take a couple of minutes depending on how many buckets you have.') ###与buckct 桶相关
        start = time.time()
        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
            setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)
            ###带attention 的seq2seq  根据seq 的id  embedding,所以 encoder input 和 decoder  input 是编序列
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, self.cell,
                    num_encoder_symbols=config.ENC_VOCAB, ###enc vocab 是编码器 字典长度
                    num_decoder_symbols=config.DEC_VOCAB,### dec vocab 是解码器 字典长度
                    embedding_size=config.HIDDEN_SIZE,###此处表示 input的 size 和 hidden size 一致，这么做方便处理，不需要做transform
                    output_projection=self.output_projection,####在ouput的分类器做transform 和projection
                    feed_previous=do_decode)###这个标志位一般是忽略decoder input 所以在测试时生效
        ####在测试阶段，需要ouput projection
        if self.fw_only:
            ###[bucket,bucket,bucket]
            ###每个bucket是list
            ###[timestep,timestep,timestep]
            ###每个time step是
            ###[batch size,output size]
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoder_inputs, 
                                        self.decoder_inputs, 
                                        self.targets,
                                        self.decoder_masks, ####需要decoder masks 和butckets
                                        config.BUCKETS, ####桶的集合，桶的集合个数影响速度？
                                        lambda x, y: _seq2seq_f(x, y, True), ####只有forward only 则不是在train阶段，所以，feed previous 为True，decoder的input由前一个时间步的decoder ouput 得到
                                        softmax_loss_function=self.softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if self.output_projection: ####若需要projection,则是对每一个桶做projection
                for bucket in range(len(config.BUCKETS)): ###对每个桶处理
                    self.outputs[bucket] = [tf.matmul(output, 
                                            self.output_projection[0]) + self.output_projection[1]
                                       for output in self.outputs[bucket]] ####对桶的每一个输出做处理
        ###在训练阶段
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoder_inputs, 
                                        self.decoder_inputs, 
                                        self.targets,
                                        self.decoder_masks,
                                        config.BUCKETS,
                                        lambda x, y: _seq2seq_f(x, y, False),
                                        softmax_loss_function=self.softmax_loss_function)
        print('Time:', time.time() - start)

    def _creat_optimizer(self):
        print('Create optimizer... \nIt might take a couple of minutes depending on how many buckets you have.')
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')####设置global step 为做lr 衰减
            ####只在训练阶段评估 optimizer，在测试阶段，不需要也没法评估
            if not self.fw_only:
                self.optimizer = tf.train.GradientDescentOptimizer(config.LR) ####采用gd，固定LR， 可以使用衰减的LR，改进点
                trainables = tf.trainable_variables()###RNN的训练一般是用clipping，所以一般用trainable variables  显式的算gradient
                self.gradient_norms = []
                self.train_ops = []
                start = time.time()
                for bucket in range(len(config.BUCKETS)):###每个bucket 分开做梯度
                    
                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket],  ###作梯度裁剪 clip
                                                                 trainables),
                                                                 config.MAX_GRAD_NORM)
                    self.gradient_norms.append(norm)
                    self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables), ###用梯度更新参数
                                                            global_step=self.global_step))
                    print('Creating opt for bucket {} took {} seconds'.format(bucket, time.time() - start))
                    start = time.time()


    def _create_summary(self):
        pass

    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._creat_optimizer()
        self._create_summary()
