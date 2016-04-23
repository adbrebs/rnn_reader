from numpy.random import choice
from numpy.random import uniform

from models.attention import EfficientAttentionModel


# Data generation
shuffle_entities = True
shuffle_questions = True
concat_ctx_and_question = False
concat_question_before = False
sort_batch_count = 20
vocab_size = 29958
n_entities = 550
batch_size = 32

# Architecture
embedding_size = uniform(100, 400)
n_hidden_que = uniform(100, 400)
n_hidden_con = n_hidden_que
n_out_hidden = uniform(100, 400)
depth_rnn = choice([1, 2, 3])
grad_clipping = 10
residual = choice([True, False])
skip_connections = choice([True, False])
bidir = choice([True, False])
dropout = choice([None, 0.1, 0.2, 0.3, 0.4, 0.5])

model = EfficientAttentionModel(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    n_hidden_que=n_hidden_que,
    n_hidden_con=n_hidden_con,
    n_out_hidden=n_out_hidden,
    n_entities=n_entities,
    depth_rnn=depth_rnn,
    grad_clipping=grad_clipping,
    residual=residual,
    bidir=bidir,
    dropout=dropout)

# Training
algo = 'adam'
learning_rate = uniform(0.0001, 0.001)
# algo = 'momentum'
# learning_rate = 0.001
max_iter = 120000
max_time = 3600 * 22

# Monitoring
train_freq_print = 100
valid_freq_print = 1000
dump_every_batches = 1000
