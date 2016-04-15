import theano
import theano.tensor as T

from model import Model1
from utilities import create_train_tag_values

floatX = theano.config.floatX = 'float32'


seq_cont = T.matrix('seq_cont', 'int32')
seq_cont_mask = T.matrix('seq_cont_mask', floatX)
seq_quest = T.matrix('seq_quest', 'int32')
seq_quest_mask = T.matrix('seq_quest_mask', floatX)
tg = T.vector('tg', 'int32')
candidates = T.matrix('candidates', 'int32')
candidates_mask = T.matrix('candidates_mask', floatX)

create_train_tag_values(seq_cont, seq_cont_mask, seq_quest, seq_quest_mask,
                        tg, candidates, candidates_mask)



model = Model1(20, 8, 7, 7, 10)

model.apply(seq_context=seq_cont,
            seq_context_mask=seq_cont_mask,
            seq_question=seq_quest,
            seq_question_mask=seq_quest_mask,
            tg=tg,
            candidates=candidates,
            candidates_mask=candidates_mask)


