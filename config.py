""" A neural chatbot using sequence to sequence model with
attentional decoder. 

See README.md for instruction on how to run the starter code.
"""
# parameters for processing the dataset
DATA_PATH = 'corpus'
CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = 'processed'
CPT_PATH = 'checkpoints'

THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 25000

BUCKETS = [(19, 19), (28, 28), (33, 33), (40, 43), (50, 53), (60, 63)] ###bucket 大小，(encoder,decoder)


CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "), 
				("don ' t ", "do n't "), ("didn ' t ", "did n't "), ("doesn ' t ", "does n't "),
				("can ' t ", "ca n't "), ("shouldn ' t ", "should n't "), ("wouldn ' t ", "would n't "),
				("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 64

LR = 0.5
MAX_GRAD_NORM = 5.0 ####用于gradient clipping

NUM_SAMPLES = 512
ENC_VOCAB = 24436
DEC_VOCAB = 24630
ENC_VOCAB = 24401
DEC_VOCAB = 24580
ENC_VOCAB = 24415
DEC_VOCAB = 24637
