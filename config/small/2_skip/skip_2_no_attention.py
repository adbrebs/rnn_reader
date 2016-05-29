from models.no_attention import GRUOneSeqModel


# Data generation
shuffle_entities = True
shuffle_questions = True
concat_ctx_and_question = True
concat_question_before = True
sort_batch_count = 20
vocab_size = 29959
n_entities = 550
batch_size = 32

# Architecture
embedding_size = 100
n_hidden = 100
n_out_hidden = 100
depth_rnn = 2
grad_clipping = 10
residual = False
skip_connections = True
bidir = False
model = GRUOneSeqModel(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    n_hidden=n_hidden,
    n_out_hidden=n_out_hidden,
    n_entities=n_entities,
    depth_rnn=depth_rnn,
    grad_clipping=grad_clipping,
    residual=residual,
    bidir=bidir)

# Training
algo = 'adam'
learning_rate = 0.001
# algo = 'momentum'
# learning_rate = 0.001

# Monitoring
train_freq_print = 100
valid_freq_print = 1000
dump_every_batches = 1000
