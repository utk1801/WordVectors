import os
import pickle
import numpy as np
from scipy import spatial

model_path = './models/'
# loss_model = 'cross_entropy'
loss_model = 'nce'
#
model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""
if loss_model == 'nce':
    outputFile = open("word_analogy_dev_predictions_nce.txt", "wb")
else:
    outputFile = open("word_analogy_dev_predictions_cross_entropy.txt", "wb")

inputFile = open("word_analogy_dev.txt", "rb")

result = ""
pair_diff = []
cosine_scores = []

for line in inputFile:
    line=line.decode() #For Python 3 issues : TypeError: a bytes-like object is required, not 'str'
    line.strip()
    example_pairs = line.split("||")[0]
    example_tuples = example_pairs.strip().split(",")
    for tuple in example_tuples:
        ex1,ex2 = tuple.strip().split(":")
        ex1 = ex1[1:]
        ex2 = ex2[:-1]
        example_embed1 = embeddings[dictionary[ex1]]
        example_embed2 = embeddings[dictionary[ex2]]
        example_diff = np.subtract(example_embed1,example_embed2)
        pair_diff.append(example_diff)

    diff_vector = np.mean(pair_diff, axis=0)

    word_pairs = line.split("||")[1]
    word_tuples = word_pairs.strip().split(",")
    cosine_scores = []
    for tup in word_tuples:
        word1, word2 = tup.strip().split(":")
        word1 = word1[1:]
        word2 = word2[:-1]
        word_embedding1 = embeddings[dictionary[word1]]
        word_embedding2 = embeddings[dictionary[word2]]
        word_diff = np.subtract(word_embedding1,word_embedding2)
        #spatial.distance.cosine computes the distance, and not the similarity.So, you must subtract the value from 1 to get the similarity.
        similarity= 1 - spatial.distance.cosine(word_diff, diff_vector)
        cosine_scores.append(similarity)

    max_idx = cosine_scores.index(max(cosine_scores))
    min_idx = cosine_scores.index(min(cosine_scores))

    result += word_pairs.strip().replace(",", "\t") + "\t" + word_tuples[max_idx].strip() + "\t" + word_tuples[min_idx] + "\n"


print("Prediction file is successfully generated.")
outputFile.write(result.encode())
outputFile.close()

##Finding Similar words for given 3 words:
print("Printing similar words for {first,american,would}: \n")
cosine_score = []
my_list=['first','american','would']
reverse_dict = dict([[v, k] for k, v in dictionary.items()])
new_dict={}
for i in range(len(my_list)):
  new_word = my_list[i]
  str_out = "Similar to %s:" % new_word
  for k,v in dictionary.items():
      top_k1 = 20  #number of similar words needed.
      # nearest_neighbour = (-sim[i,:]).argsort()[1:top_k1 + 1]
      # print(v)
      dict_embedding = embeddings[v]
      word_embedding = embeddings[dictionary[new_word]]
      similarity = 1 - spatial.distance.cosine(dict_embedding, word_embedding)
      cosine_score.append(similarity)
      new_dict[similarity] = k
  top_20_idx = np.argsort(cosine_score)[-20:]
  top_20_values = [cosine_score[i] for i in top_20_idx]
  for idx in range(0,top_k1,1):
    similar_words = new_dict[top_20_values[idx]]
    str_out = "%s %s," % (str_out,similar_words)
  print(str_out)