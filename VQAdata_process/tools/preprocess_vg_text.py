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
import stanfordcorenlp
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
from tqdm import tqdm

try:
    import cPickle as pickle
except:
    import pickle

print('load nlp model...')
nlp = StanfordCoreNLP(r'/hdd1/caojianjian/stanford_corenlp_master/stanford-corenlp-full-2018-10-05')
print("nlp model load over!")

contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":"could've", "couldnt": "couldn't", "couldn'tve": "couldn't've","couldnt've": "couldn't've", "didnt": "didn't", "doesnt":"doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":"hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":"haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":"he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll","hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":"I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":"it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's","maam": "ma'am", "mightnt": "mightn't", "mightnt've":"mightn't've", "mightn'tve": "mightn't've", "mightve": "might've","mustnt": "mustn't", "mustve": "must've", "neednt": "needn't","notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't","ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":"'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":"she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":"shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":"shouldn't've", "somebody'd": "somebodyd", "somebodyd've":"somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":"somebody'll", "somebodys": "somebody's", "someoned": "someone'd","someoned've": "someone'd've", "someone'dve": "someone'd've","someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", "something'dve":"something'd've", "somethingll": "something'll", "thats":"that's", "thered": "there'd", "thered've": "there'd've","there'dve": "there'd've", "therere": "there're", "theres":"there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":"they'd've", "theyll": "they'll", "theyre": "they're", "theyve":"they've", "twas": "'twas", "wasnt": "wasn't", "wed've":"we'd've", "we'dve": "we'd've", "weve": "we've", "werent":"weren't", "whatll": "what'll", "whatre": "what're", "whats":"what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod":"who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll","whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":"would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've","wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":"y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've","y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":"you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":"you'll", "youre": "you're", "youve": "you've"}

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
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer

def process_answers(q):
    print("Process VG answer ...")
    answers_dict = pickle.load(open('../VQA/question_graph/train_data/vqa_trainval_a_dict.p', 'rb'))
    ans_wtoi = answers_dict['wtoi']
    vocab = []
    for word, i in ans_wtoi.items():
        vocab.append(word)

    vg_trainval_data = []
    for row in tqdm(q):
        accepted_answers = 0
        for w, c in row['answers']:
            if w in vocab:
                accepted_answers += c

        if accepted_answers != 0:
            answers_scores = []
            for w, c in row['answers']:
                if w in vocab:
                    answers_scores.append((w, c / accepted_answers))
            row['answers_w_scores'] = answers_scores
            vg_trainval_data.append(row)

    print('vg trainval questions number:', len(vg_trainval_data))

    return vg_trainval_data


def process_questions(q, phase):
    print('process question and build question vocab...')
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
    pickle.dump({'itow': itow, 'wtoi': wtoi}, open('../VQA/question_graph/trainval_data/vqa_' + phase + '_q_dict.p', 'wb'))

def build_question_graph(qa, phase):
    print('Building VG question graph...')
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
    json.dump(question_graph_dict, open('../VQA/question_graph/trainval_data/vg_aug_'+ phase +'_q_graph.json', 'w'))

def process_vg(vgq, _vgv, img_id2val, ans2label, phase):
    # Combine questions and answers and build question graph
    print('Process VG question answer data...')
    data = []
    vgv = {}
    for _v in _vgv:
        if _v['coco_id']:
            vgv[_v['image_id']] = _v['coco_id']
    # used image, used question, total question, out-of-split
    counts = [0, 0, 0, 0]
    for vg in vgq:
        coco_id = vgv.get(vg['id'], None)
        if coco_id is not None:
            counts[0] += 1
            for q in vg['qas']:
                if q['answer'].lower() in ['yes','no','yes.','no.']:
                    print(q['answer'])
            img_idx = img_id2val.get(coco_id, None)
            if img_idx is None:
                counts[3] += 1
            for q in vg['qas']:
                counts[2] += 1
                _answer = preprocess_answer(q['answer'])
                label = ans2label.get(_answer, None)
                if label and img_idx:
                    counts[1] += 1
                    row = {
                            'question_id': q['qa_id'],
                            'image_id': coco_id,
                            'image': img_idx,
                            'question': q['question'],
                            'answer': _answer,
                            'answers': [(_answer, 10)],
                            'answers_w_scores': [(_answer, 1.0)]}
                    data.append(row)
    print('Loading VisualGenome %s' % phase)
    print('\tUsed COCO images: %d/%d (%.4f)' %
            (counts[0], len(_vgv), counts[0]/len(_vgv)))
    print('\tOut-of-split COCO images: %d/%d (%.4f)' %
            (counts[3], counts[0], counts[3]/counts[0]))
    print('\tUsed VG questions: %d/%d (%.4f)' %
            (counts[1], counts[2], counts[1]/counts[2]))

    ## build question graph
    build_question_graph(data, phase)

if __name__ == '__main__':

    # train data
    print('processing visual genome data...')
    qas = json.load(open('./VG/question_answers.json'))
    idx = json.load(open('./VG/image_data.json'))
    ans2label = pickle.load(open('../VQA/question_graph/train_data/vqa_trainval_a_dict.p', 'rb'))['wtoi']
    img_id2train = pickle.load(open('./VG/imgids/train_imgid2idx.pkl','rb'))
    process_vg(qas, idx, img_id2train, ans2label, 'train')

    img_id2val = pickle.load(open('./VG/imgids/val_imgid2idx.pkl','rb'))
    process_vg(qas, idx, img_id2val, ans2label, 'val')

    vg_q_graph = json.load(open('../VQA/question_graph/trainval_data/vg_aug_train_q_graph.json')) + json.load(open('../VQA/question_graph/trainval_data/vg_aug_val_q_graph.json'))
    question_graph_dict = process_answers(vg_q_graph)
    json.dump(question_graph_dict, open('../VQA/question_graph/trainval_data/vg_aug_q_graph.json', 'w'))

    question_graph_all = json.load(open('../VQA/question_graph/train_data/vqa_train_q_graph.json')) + json.load(open('../VQA/question_graph/train_data/vqa_val_q_graph.json')) + json.load(open('../VQA/question_graph/trainval_data/vqa_test_q_graph.json')) + json.load(open('../VQA/question_graph/trainval_data/vg_aug_q_graph.json'))
    process_questions(question_graph_all, 'all_aug')
    print('Done')