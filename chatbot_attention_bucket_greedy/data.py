""" A neural chatbot using sequence to sequence model with
attentional decoder. 
See readme.md for instruction on how to run the starter code.
"""
import os
import random
import re

import numpy as np

import config

def get_lines():
    id2line = {}
    file_path = os.path.join(config.DATA_PATH, config.LINE_FILE)
    print(config.LINE_FILE)
    with open(file_path, 'r', errors='ignore') as f:
        # lines = f.readlines()
        # for line in lines:
        i = 0
        try:
            for line in f:
                parts = line.split(' +++$+++ ')
                if len(parts) == 5:
                    if parts[4][-1] == '\n':###如果行尾含\n,去掉\n
                        parts[4] = parts[4][:-1]
                    id2line[parts[0]] = parts[4]
                i += 1
        except UnicodeDecodeError:
            print(i, line)
    return id2line

def get_convos():
    """ Get conversations from the raw data """
    file_path = os.path.join(config.DATA_PATH, config.CONVO_FILE)
    convos = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.split(' +++$+++ ')
            if len(parts) == 4:
                convo = []
                for line in parts[3][1:-2].split(', '):###[1:-2] 去掉回车和]
                    convo.append(line[1:-1])###去掉''
                convos.append(convo)

    return convos

def question_answers(id2line, convos):
    """ Divide the dataset into two sets: questions and answers. """
    questions, answers = [], []
    for convo in convos:
        for index, line in enumerate(convo[:-1]):###convo[:-1]不包括最后一个item
            questions.append(id2line[convo[index]])####对话的question 和answer是相互的
            answers.append(id2line[convo[index + 1]])
    assert len(questions) == len(answers)
    # for i in range(len(questions)):
    #     print('\n'+'Q: '+questions[i],'A: '+answers[i]+'\n')
    return questions, answers

def prepare_dataset(questions, answers):
    # create path to store all the train & test encoder & decoder
    make_dir(config.PROCESSED_PATH)
    
    # random convos to create the test set
    test_ids = random.sample([i for i in range(len(questions))],config.TESTSET_SIZE) ####随机选出TESTSET_SIZE个测试数据

    filenames = ['train.enc', 'train.dec', 'test.enc', 'test.dec'] ####训练encoder 训练decoder，测试encoder 测试decoder
    files = []
    for filename in filenames:
        files.append(open(os.path.join(config.PROCESSED_PATH, filename),'w'))

    for i in range(len(questions)):
        if i in test_ids:
            files[2].write(questions[i] + '\n')
            files[3].write(answers[i] + '\n')
        else:
            files[0].write(questions[i] + '\n')
            files[1].write(answers[i] + '\n')

    for file in files:
        file.close()

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass
####对每一行语句进行基本的处理，去掉无用的符号
def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")####delimiter 加上() 表示保留分割符
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token) ####数字替换为#
            words.append(token)
    return words

def build_vocab(filename, normalize_digits=True):
    in_path = os.path.join(config.PROCESSED_PATH, filename)
    out_path = os.path.join(config.PROCESSED_PATH, 'vocab.{}'.format(filename[-3:]))

    vocab = {}####单词计数
    with open(in_path, 'r') as f:
        for line in f.readlines():
            for token in basic_tokenizer(line):
                if not token in vocab:
                    vocab[token] = 0
                vocab[token] += 1

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True) ####排序
    with open(out_path, 'w') as f:
        f.write('<pad>' + '\n')###先填充4个特定符号
        f.write('<unk>' + '\n')
        f.write('<s>' + '\n')###句子开始
        f.write('<\s>' + '\n')###句子结尾
        index = 4
        for word in sorted_vocab:
            if vocab[word] < config.THRESHOLD: ####把出现频率小于THRESHOLD的单词去掉，因为是逆序排序，遇到小于THRESHOLD就退出循环
                break
            f.write(word + '\n')
            index += 1
        with open('config.py', 'a') as cf:
            if filename[-3:] == 'enc':
                cf.write('ENC_VOCAB = ' + str(index) + '\n')
            else:
                cf.write('DEC_VOCAB = ' + str(index) + '\n')

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))} ####返回单词，单词：编号 集合

def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)]####vocab.get  如果取不到，则返回<unk>的编号

def token2id(data, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    _, vocab = load_vocab(os.path.join(config.PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(config.PROCESSED_PATH, in_path), 'r')
    out_file = open(os.path.join(config.PROCESSED_PATH, out_path), 'w')
    
    lines = in_file.read().splitlines()
    for line in lines:
        if mode == 'dec': # we only care about '<s>' and </s> in decoder 在decoder中，加入句子开头和结束符
            ids = [vocab['<s>']]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))
        # ids.extend([vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)])
        if mode == 'dec':
            ids.append(vocab['<\s>'])
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')

def prepare_raw_data():
    print('Preparing raw data into train set and test set ...')
    id2line = get_lines()
    convos = get_convos()
    questions, answers = question_answers(id2line, convos)
    prepare_dataset(questions, answers)

def process_data():
    print('Preparing data to be model-ready ...')
    build_vocab('train.enc') ####字典是从train encoder 和 train decoder 得到，test的语料没有在vocab中，故train中未见到的词在test中都为<unk>
    build_vocab('train.dec')
    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')

### 读取encoder，decoder 数据
def load_data(enc_filename, dec_filename, max_training_size=None):
    encode_file = open(os.path.join(config.PROCESSED_PATH, enc_filename), 'r')
    decode_file = open(os.path.join(config.PROCESSED_PATH, dec_filename), 'r')
    encode, decode = encode_file.readline(), decode_file.readline()####读取一行读取数据
    data_buckets = [[] for _ in config.BUCKETS]
    i = 0
    ####对所有数据分buckets，将类似的句子长度分到同一个bucket中，
    while encode and decode:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i)
        encode_ids = [int(id_) for id_ in encode.split()]
        decode_ids = [int(id_) for id_ in decode.split()]
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size: ###将句子的encoder 和decoder长度都小于设定的bucket，就放入这个bucket中
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break ####每行数据只往一个bucket中放置，
        encode, decode = encode_file.readline(), decode_file.readline() ###再次读取一行数据
        i += 1
    return data_buckets

def _pad_input(input_, size):
    return input_ + [config.PAD_ID] * (size - len(input_))####对数据padding，使得句子长度一致

def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    ### encoder_size 就是 time step[[],[],[],[],[],[],[],[]] 长度为time step，每个[]中是[,,,,,,]长度为batch size
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                    for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs


def get_batch(data_bucket, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    encoder_size, decoder_size = config.BUCKETS[bucket_id]####按桶id 取 encoder size 和decoder size 取得当前桶的encoder size 和decoder size
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size): ###取得batch size 条数据 并做padding
        encoder_input, decoder_input = random.choice(data_bucket) ####随机选取桶中的一条记录
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size)))) ####encoder是reverse的，把qestion和answerpadding 到和桶的size 一样大小
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))

    # now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)####encoder size 同时也是桶的一行数据的size，batch size 是批量数据
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)###对batch 做reshape

    # create decoder_masks to be 0 for decoders that are padding. decoder mask
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32) ###batch mask是一维的
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == config.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks

if __name__ == '__main__':
    prepare_raw_data()
    process_data()
