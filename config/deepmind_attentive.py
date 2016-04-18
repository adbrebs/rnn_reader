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
# sequence processing
embedding_size = 200
n_hidden_que = 256
n_hidden_con = 256
# attention
n_attention = 100
# output
n_out_hidden = 100

model = SoftmaxAttentionModel(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    n_hidden_que=n_hidden_que,
    n_hidden_con=n_hidden_con,
    n_entities=n_entities,
    n_attention=n_attention,
    n_out_hidden=n_out_hidden)

# Training
algo = 'adam'
learning_rate = 0.1

# Monitoring
train_freq_print = 100
valid_freq_print = 1000
