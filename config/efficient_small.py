from models.efficient import Model


# Data generation
shuffle_entities = True
shuffle_questions = True
concat_ctx_and_question = False
concat_question_before = False
sort_batch_count = 20
vocab_size = 29958
n_entities = 550
batch_size = 16

# Architecture
embedding_size = 100
n_hidden_quest = 100
n_hidden_cont = 100
model = Model(vocab_size=vocab_size,
              embedding_size=embedding_size,
              n_hidden_question=n_hidden_quest,
              n_hidden_context=n_hidden_cont,
              n_entities=n_entities)

# Trianing
algo = 'adam'
learning_rate = 0.1

# Monitoring
train_freq_print = 100
valid_freq_print = 1000
