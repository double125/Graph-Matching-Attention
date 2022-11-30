from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import collections
import argparse
import string
from spacy.tokenizer import Tokenizer
import re
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

print('load nlp model...')
nlp = StanfordCoreNLP(r'stanford_corenlp_master/stanford_corenlp_full_2018_10_05')
print("nlp model load over!")

# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))

manual_map = { 'none': '0',
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10'}

articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
        '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!']

def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
                or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText

def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    outText = ' '.join(outText)
    return outText

def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer

def filter_answers(answers_dset, min_occurence):
    """ This will change the answer to preprocessed version"""
    occurence = {}
    for ans_entry in answers_dset:
        answers = ans_entry['answers']
        gtruth = ans_entry['multiple_choice_answer']
        gtruth = preprocess_answer(gtruth)
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(ans_entry['question_id'])

    for answer in list(occurence):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)
    print('Num of answers that appear >= %d times: %d' % (min_occurence, len(occurence)))

    print('Build final output dictionary...')
    itow = {}
    wtoi = {}
    label = 0
    for answer in occurence:
        itow[label] = answer
        wtoi[answer] = label
        label += 1

    pickle.dump({'itow': itow, 'wtoi': wtoi}, open('../VQA/question_graph/train_data/vqa_trainval_a_dict.p', 'wb'))

def get_score(occurences):
    if occurences == 0:
        return .0
    elif occurences == 1:
        return .3
    elif occurences == 2:
        return .6
    elif occurences == 3:
        return .9
    else:
        return 1.

def process_answers(q, phase):
    answers_dict = pickle.load(open('../VQA/question_graph/train_data/vqa_trainval_a_dict.p', 'rb'))
    ans_wtoi = answers_dict['wtoi']
    vocab = []
    for word, i in ans_wtoi.items():
        vocab.append(word)

    for row in tqdm(q):
        answers = []
        for w, c in row['answers']:
            w = preprocess_answer(w)
            #w = process_punctuation(w)
            answers.append([w, c])
        row['answers'] = answers
        #row['answer'] = process_punctuation(row['answer'])
        row['answer'] = preprocess_answer(row['answer'])

        answers_scores = []
        for w, c in row['answers']:
            if w in vocab:
                answers_scores.append((w, get_score(c)))
        row['answers_w_scores'] = answers_scores

    json.dump(q, open('../VQA/question_graph/train_data/vqa_' + phase + '_q_graph.json', 'w'))


def process_questions(q, phase):
    print('process question and build question vocab...')
    # build question dictionary
    def build_vocab(questions):
        count_thr = 1
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
    pickle.dump({'itow': itow, 'wtoi': wtoi}, open('../VQA/question_graph/train_data/vqa_' + phase + '_q_dict2.p', 'wb'))

def build_question_graph(qa, phase):
    print('Building ' + phase +' question graph...')
    qas = len(qa)
    question_graph_dict = []
    for i, item in enumerate(tqdm(qa)):
        question = item['question'].lower().replace('/',' / ').replace('-',' ').replace('?',' ').replace('.',' ').strip(' ')  # question graph 1-4

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
    if phase == 'test' or phase == 'testdev':
        json.dump(question_graph_dict, open('../VQA/question_graph/trainval_data/vqa_' + phase + '_q_graph.json','w'))
    else:
        return question_graph_dict

def process_qa(questions, annotations, phase):
    # Combine questions and answers and build question graph
    print('Process ' + phase +' question answer data...')
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
        data.append(row)

    # build question graph
    question_graph = build_question_graph(data, phase)

    # process answers
    process_answers(question_graph, phase)


if __name__ == '__main__':

    # Combine Q and A
    print('Combining question and answer...')
    ## train data
    train_questions = json.load(open('raw/v2_OpenEnded_mscoco_train2014_questions.json'))
    train_answers = json.load(open('raw/v2_mscoco_train2014_annotations.json'))['annotations']

    ## val data
    val_questions = json.load(open('raw/v2_OpenEnded_mscoco_val2014_questions.json'))
    val_answers = json.load(open('raw/v2_mscoco_val2014_annotations.json'))['annotations']

    ## test data
    testdev_questions = json.load(open('raw/v2_OpenEnded_mscoco_test-dev2015_questions.json'))
    test_questions = json.load(open('raw/v2_OpenEnded_mscoco_test2015_questions.json'))

    ## build the final answer dictionary
    answers = train_answers + val_answers
    filter_answers(answers, 9)

    ## process trainval data
    process_qa(train_questions, train_answers, 'train')
    process_qa(val_questions, val_answers, 'val')
    ## process test data
    build_question_graph(test_questions['questions'], 'test')
    build_question_graph(testdev_questions['questions'], 'testdev')

    question_graph_all = json.load(open('../VQA/question_graph/train_data/vqa_train_q_graph.json')) + json.load(open('../VQA/question_graph/train_data/vqa_val_q_graph.json')) + json.load(open('../VQA/question_graph/trainval_data/vqa_test_q_graph.json')) + json.load(open('../VQA/question_graph/trainval_data/vqa_testdev_q_graph.json'))
    process_questions(question_graph_all, 'all')
    print('Done')
