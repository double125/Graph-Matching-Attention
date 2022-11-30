from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import collections
import argparse
import string
from spacy.tokenizer import Tokenizer
import en_core_web_sm

import sys
sys.path.append("..")
import stanford_corenlp_master
from stanford_corenlp_master import stanfordcorenlp
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
from tqdm import tqdm

try:
    import cPickle as pickle
except:
    import pickle

nlp = en_core_web_sm.load()
tokenizer = Tokenizer(nlp.vocab)
exclude = set(string.punctuation)


print('load nlp model...')
nlp = StanfordCoreNLP(r'stanford_corenlp_master/stanford_corenlp_full_2018_10_05')
print("nlp model load over!")


def process_answers(q, phase, n_answers=3000):

    # find the n_answers most common answers
    counts = {}
    for row in q:
        counts[row['answer']] = counts.get(row['answer'], 0) + 1

    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print("total answer:",len(cw))

    vocab = [w for c, w in cw[:n_answers]]

    # a 0-indexed vocabulary translation table
    itow = {i: w for i, w in enumerate(vocab)}
    wtoi = {w: i for i, w in enumerate(vocab)}  # inverse table
    pickle.dump({'itow': itow, 'wtoi': wtoi}, open(phase + '_a_dict.p', 'wb'))

    for row in q:
        accepted_answers = 0
        for w, c in row['answers']:
            if w in vocab:
                accepted_answers += c

        answers_scores = []
        for w, c in row['answers']:
            if w in vocab:
                answers_scores.append((w, c / accepted_answers))

        row['answers_w_scores'] = answers_scores

    json.dump(q, open('vqa_' + phase + '_final_{}_q_graph.json'.format(n_answers), 'w'))


def process_questions(q):
    # build question dictionary
    def build_vocab(questions):
        count_thr = 0
        # count up the number of times a word is used
        counts = {}
        for row in questions:
            for word in row['question_toked']:
                counts[word] = counts.get(word, 0) + 1
        cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
        print('top words and their counts:')
        print('\n'.join(map(str, cw[:10])))

        # print some stats
        total_words = sum(counts.values())
        print('total words:', total_words)
        bad_words = [w for w, n in counts.items() if n <= count_thr]
        vocab = [w for w, n in counts.items() if n > count_thr]
        bad_count = sum(counts[w] for w in bad_words)
        print('number of bad words: %d/%d = %.2f%%' %
              (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
        print('number of words in vocab would be %d' % (len(vocab), ))
        print('number of UNKs: %d/%d = %.2f%%' %
              (bad_count, total_words, bad_count*100.0/total_words))

        return vocab

    vocab = build_vocab(q)
    # a 1-indexed vocab translation table
    itow = {i+1: w for i, w in enumerate(vocab)}
    wtoi = {w: i+1 for i, w in enumerate(vocab)}  # inverse table
    pickle.dump({'itow': itow, 'wtoi': wtoi}, open(phase + '_q_dict.p', 'wb'))


def tokenize_questions(qa, phase):
    qas = len(qa)
    for i, row in enumerate(tqdm(qa)):
        row['question_toked'] = [t.text if '?' not in t.text else t.text[:-1]
                                 for t in tokenizer(row['question'].lower())]  # get spacey tokens and remove question marks
        if i == qas - 1:
            json.dump(qa, open('vqa_' + phase + '_toked.json', 'w'))

def build_question_graph(qa, phase):
    qas = len(qa)
    question_graph_dict = []
    for i, item in enumerate(tqdm(qa)):
        question = item['question'].lower().replace('/',' / ').replace('-',' ').replace('?',' ').replace('.',' ').strip(' ')  # question graph 1-4
        #question = item['question'].lower().strip()

        # question Tokenizer processing
        q_toked = nlp.word_tokenize(question)
        q_toked = [q.lower() for q in q_toked]

        item['question_toked'] = q_toked

        # dependency_parse processing
        dp_dict = nlp.dependency_parse(question)[1:]

        node_num = len(q_toked)
        A_Matrix = np.eye(node_num, dtype=int)
        for dp in dp_dict:
            i, j = dp[1]-1, dp[2] - 1 # dependency_parse index strat from 1
            #A_Matrix[i][j] = A_Matrix[j][i] = 1    #question graph 4
            A_Matrix[i][j] = 1     #question graph 5
        item['question_A_Matrix'] = A_Matrix.tolist()

        # question_parser_graph_nodes construct
        question_parser_graph_nodes = {}
        for row in range(node_num):
            col = np.nonzero(A_Matrix[row])[0]
            node = [q_toked[i] for i in col.tolist()]
            question_parser_graph_nodes[row] = node
        item['question_parser_graph_nodes'] = question_parser_graph_nodes

        # question_toked_graph_nodes construct
        question_toked_graph_nodes = {}
        for i,tokens in enumerate(q_toked):
            question_toked_graph_nodes[i] = tokens
        item['question_toked_graph_nodes'] = question_toked_graph_nodes
        #print(item)

        question_graph_dict.append(item)

    json.dump(question_graph_dict, open('vqa_' + phase + '_toked_q_graph.json','w'))

def combine_qa(questions, annotations, phase):
    # Combine questions and answers in the same json file
    # 443757 questions
    data = []
    for i, q in enumerate(tqdm(questions['questions'])):
        row = {}
        # load questions info
        row['question'] = q['question']
        row['question_id'] = q['question_id']
        row['image_id'] = str(q['image_id'])

        # load answers
        assert q['question_id'] == annotations[i]['question_id']
        row['answer'] = annotations[i]['multiple_choice_answer']

        answers = []
        for ans in annotations[i]['answers']:
            answers.append(ans['answer'])
        row['answers'] = collections.Counter(answers).most_common()
        #print(row)

        data.append(row)

    json.dump(data, open('vqa_' + phase + '_combined.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description='Preprocessing for VQA v2 text data')
    parser.add_argument('--data', nargs='+', help='train, val and/or test, list of data phases to be processed', required=True)
    parser.add_argument('--nanswers', default=3000, type=int, help='number of top answers to consider for classification.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise SystemExit('Unknown argument: {}'.format(unparsed))

    phase_list = args.data
    print(args.nanswers)

    for phase in phase_list:

        print('processing ' + phase + ' data')
        if phase != 'test':
            # Combine Q and A
            print('Combining question and answer...')
            question = json.load(
                open('raw/v2_OpenEnded_mscoco_' + phase + '2014_questions.json'))
            answers = json.load(open('raw/v2_mscoco_' + phase + '2014_annotations.json'))
            combine_qa(question, answers['annotations'], phase)

        #    # Tokenize
            print('Building question graph...')
            t = json.load(open('vqa_' + phase + '_combined.json'))
            build_question_graph(t, phase)
        else:
            print ('Building question graph...')
            t = json.load(open('raw/v2_OpenEnded_mscoco_' + phase + '2015_questions.json'))
            t = t['questions']
            build_question_graph(t, phase)

        # Build dictionary for question and answers
        print('Building dictionary...')
        t = json.load(open('vqa_' + phase + '_toked_q_graph.json'))
        if phase == 'train':
            process_questions(t)
        if phase != 'test':
            process_answers(t, phase, n_answers=args.nanswers)

    print('Done')
