from models.no_attention import GRUModel


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
embedding_size = 256
n_hidden = 256
n_out_hidden = 256
depth_rnn = 1
grad_clipping = 10
residual = False
skip_connections = False
bidir = False
model = GRUModel(
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
