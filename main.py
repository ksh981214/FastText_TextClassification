import torch
from random import shuffle
from collections import Counter
import argparse
import random
import time
import numpy as np
import string

def fnv(s, bucket):
    prime = 0x100000001b3
    basis = 0xcbf29ce484222325
    hashing = basis
    
    for cash in s:
        hashing = hashing^ord(cash)
        hashing = (hashing * prime) % (2**64)
    return hashing % bucket

def word2vec_trainer(labels,idx_titles, idx_descriptions, word2ind, gram2hashed, ind2hashed, hashed2ind, dimension, learning_rate):
    # Xavier initialization of weight matrices
    class_num = 4
    W_emb = torch.randn(len(word2ind)+ len(hashed2ind), dimension) / (dimension**0.5) 
    W_out = torch.randn(class_num, dimension) / (dimension**0.5)  

    epoch_num = 5
    iters = 0
    epoch_size = len(labels) 
    total_iter = epoch_num * epoch_size
    
    initial_lr = learning_rate

    losses=[]
    for epoch in range(epoch_num):
        for i in range(epoch_size):
            iters += 1
            
            learning_rate = initial_lr * (1-iters/total_iter)

            #Training word2vec using SGD
            label=labels[i]
            title=idx_titles[i]
            description=idx_descriptions[i]

            all_idx = title+description
            vec_all = torch.mean(W_emb[all_idx, :],dim=0,keepdim=True)
            score_vector = torch.matmul(vec_all, torch.t(W_out)) # (1,D) * (D,C) = (1,C)
            e = torch.exp(score_vector) 

            softmax = e / (torch.sum(e, dim=1, keepdim=True)) #1,C
            loss = -np.log(softmax[:,label] + 1e-7)

            #get grad
            softmax_grad = softmax
            softmax_grad[:,label] -= 1.0

            grad_out = torch.matmul(torch.t(softmax_grad), vec_all) #(C, 1) * (1,D) = (C,D)
            grad_emb = torch.matmul(softmax_grad, W_out) #(1,C) * (C,D) = (1,D)

            W_emb[all_idx] -= learning_rate*grad_emb
            W_out -= learning_rate*grad_out
            losses.append(loss)

            if i%10000==0:
                avg_loss=sum(losses)/len(losses)
                print("epoch: %d, i: %d, Loss : %f, learning_rate : %f " %(epoch, i, avg_loss, learning_rate))
                losses=[]

    return W_emb, W_out

def main():
	#Load and tokenize corpus
    print("loading...")
    start = time.time()
    train = open("ag_news_csv/train.csv").readlines()
    datas = train
    #len(datas) #120000
    
    labels=[]
    titles=[]
    descriptions=[]
    for data in datas: 
        temp = data.lower().split("\",\"")
        for c in temp[0]:
            if c in string.digits:
                labels.append(int(c)-1)

        title =''
        for c in temp[1]:
            if c in string.ascii_lowercase or c == ' ':
                title+=c
            elif c =="\\":
                title = title + ' '
        title = " ".join(title.split())
        titles.append(title)

        description =''
        for c in temp[2]:
            if c in string.ascii_lowercase or c == ' ':
                description+=c
            elif c =="\\":
                description = description + " "
        description = " ".join(description.split())
        descriptions.append(description)

    # print(len(labels))
    # print(len(titles))
    # print(len(descriptions))

    print("Title tokenizing...")
    #Assign an index number to a word
    word2ind = {}
    word2ind[" "]=0
    i = 1
    for title in titles:
        for word in title.split():
            if word not in word2ind.keys():
                word2ind[word] = i
                i+=1
    print("Description tokenizing...")
    for description in descriptions:
        for word in description.split():
            if word not in word2ind.keys():
                word2ind[word] = i
                i+=1

    ind2word = {}
    for k,v in word2ind.items():
        ind2word[v]=k

    print("Word Vocabulary size")
    print(len(word2ind))
    print()
    feed_dict = {}

    print("Making Bi-gram ....")
    #make n-gram
    gram2hashed = {}
    ind2hashed = {}
    hashed2ind = {}
    
    num_bigram = 0

    idx_titles=[]
    for title in titles:
        idx_title=[]
        temp = title.split()
        length = len(temp)
        bigram = ''
        for idx,word in enumerate(temp):
            if idx == length-1:
                break
            else:
                bigram = word +"-"+temp[idx+1]
                num_bigram += 1
            hashed = fnv(bigram, 2100000)
            if bigram not in gram2hashed.keys():
                gram2hashed[bigram]= hashed
            if hashed not in hashed2ind.keys():
                ind2hashed[i]=hashed
                hashed2ind[hashed]= i
                i += 1
            idx_title.append(word2ind[word])
            idx_title.append(hashed2ind[hashed]) 
        idx_titles.append(idx_title)

    idx_descriptions=[]
    for description in descriptions:
        idx_description=[]
        temp = description.split()
        length = len(temp)
        bi_gram = ''
        for idx,word in enumerate(temp):
            if idx == length-1:
                break
            else:
                bigram = word +"-"+temp[idx+1]
                num_bigram += 1
            hashed = fnv(bigram, 2100000)
            if bigram not in gram2hashed.keys():
                gram2hashed[bigram]= hashed
            if hashed not in hashed2ind.keys():
                ind2hashed[i]=hashed
                hashed2ind[hashed]= i
                i += 1 
            idx_description.append(word2ind[word])
            idx_description.append(hashed2ind[hashed])
        idx_descriptions.append(idx_description)

    print("len(idx_titles, idx_descriptions):{},{}".format(len(idx_titles), len(idx_descriptions)))

    print("num_bigram : {}".format(num_bigram))
    print("Preprocess consume time: {}".format(time.time()-start))
    print("bigrams hashed size: {}".format(len(hashed2ind)))
    print("Total WordEmbedding size: {}".format(len(word2ind) + len(hashed2ind)))
    print()
    
    #Training section
    print("Training Start...")
    start = time.time()
    #learning_rate=[0.05, 0.1. 0.25, 0.5]
    W_emb,W_out = word2vec_trainer(labels,idx_titles,idx_descriptions, word2ind, gram2hashed, ind2hashed, hashed2ind, dimension=10, learning_rate=0.25)
    print("Training Consume time: {}".format(time.time()-start))

    print("Testing Start...")
    start = time.time()
    test = open("ag_news_csv/test.csv").readlines()
    datas = test
    #len(datas) #120000
    
    test_labels=[]
    test_titles=[]
    test_descriptions=[]
    for data in datas: 
        temp = data.lower().split("\",\"")
        for c in temp[0]:
            if c in string.digits:
                test_labels.append(int(c)-1)

        title =''
        for c in temp[1]:
            if c in string.ascii_lowercase or c == ' ':
                title+=c
            elif c =="\\":
                title = title + ' '
        title = " ".join(title.split())
        test_titles.append(title)

        description =''
        for c in temp[2]:
            if c in string.ascii_lowercase or c == ' ':
                description+=c
            elif c =="\\":
                description = description + " "
        description = " ".join(description.split())
        test_descriptions.append(description)

    test_idx_titles=[]
    for title in test_titles:
        idx_title=[]
        temp = title.split()
        length = len(temp)
        bigram = ''
        for idx,word in enumerate(temp):
            if idx == length-1:
                break
            else:
                bigram = word +"-"+temp[idx+1]
            if word in word2ind.keys():
                idx_title.append(word2ind[word])
            if bigram in gram2hashed.keys():
                idx_title.append(hashed2ind[gram2hashed[bigram]]) 
        test_idx_titles.append(idx_title)

    test_idx_descriptions=[]
    for description in test_descriptions:
        idx_description=[]
        temp = description.split()
        length = len(temp)
        bi_gram = ''
        for idx,word in enumerate(temp):
            if idx == length-1:
                break
            else:
                bigram = word +"-"+temp[idx+1]
            if word in word2ind.keys():
                idx_description.append(word2ind[word])
            if bigram in gram2hashed.keys():
                idx_description.append(hashed2ind[gram2hashed[bigram]])
        test_idx_descriptions.append(idx_description)
    
    preds=[]
    for i, (title, description) in enumerate(zip(test_idx_titles, test_idx_descriptions)):
        all_idx = title+description
        vec_all = torch.mean(W_emb[all_idx, :],dim=0,keepdim=True)
        score_vector = torch.matmul(vec_all, torch.t(W_out)) # (1,D) * (D,C) = (1,C)
        preds.append(torch.argmax(score_vector,1))
    
    total = len(preds)
    corr = 0
    result_txt = open("result.txt","w")
    for i,pred in enumerate(preds):
        if test_labels[i] == pred:
            corr += 1
            
        result_txt.write(str((pred+1).item())+"\n")
    
    total_result = "Correct Answer/Total: {}/{}".format(corr, total)
    result_txt.write(total_result)
    result_txt.close()
    
    print(total_result)

    
main()
