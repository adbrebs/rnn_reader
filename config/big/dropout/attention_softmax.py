from models.attention import SoftmaxAttentionModel


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
embedding_size = 256
n_hidden_que = 256
n_hidden_con = 256
n_attention = 256
n_out_hidden = 256
depth_rnn = 1
grad_clipping = 10
residual = False
skip_connections = False
bidir = False
dropout = 0.3

model = SoftmaxAttentionModel(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    n_hidden_que=n_hidden_que,
    n_hidden_con=n_hidden_con,
    n_entities=n_entities,
    n_attention=n_attention,
    n_out_hidden=n_out_hidden,
    depth_rnn=depth_rnn,
    grad_clipping=grad_clipping,
    residual=residual,
    bidir=bidir,
    dropout=dropout)

# Training
algo = 'adam'
learning_rate = 0.0003

# Monitoring
train_freq_print = 100
valid_freq_print = 1000
dump_every_batches = 10000
