import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']

em1 = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")

sentence_embeddings1 = em1.embed_documents(sentences)
sentence_embeddings1 = np.array(sentence_embeddings1)

print("sentence_embeddings1 shape=", sentence_embeddings1.shape)
vec1, vec2 = sentence_embeddings1
cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print("余弦相似度：%.3f" % cos_sim)
