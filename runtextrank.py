import logging
import pytextrank
import spacy
import sys
import os

nlp = spacy.load("en_core_web_sm")

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("PyTR")


from spacy.language import Language
@Language.component("textrank")
def textrank(doc):
    tr = pytextrank.TextRank()
    doc = tr.PipelineComponent(doc)
    return doc
nlp.add_pipe("textrank", last=True)

#tr = pytextrank.TextRank(logger=None)
#nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)


from datetime import datetime
startTime = datetime.now()


for i in range (1, 4713):
    # read source documents
    reviewlist = []
    for dir2 in os.listdir(f'./data/topic_{i}/input_docs'):
        currfile = open(f'./data/topic_{i}/input_docs/{dir2}')
        temp = ''.join(currfile.readlines())
        temp = temp.replace('\n', ' ')
        temp = temp.replace('\t', ' ')
        reviewlist.append(temp)
        break
        # print(temp)
    text = ""
    for x in reviewlist:
    	text = text + x;
    doc = nlp(text)
    result = ""
    for sent in doc._.textrank.summary(limit_phrases=30, limit_sentences=10):
    	result = result + sent.text + '. '
    	# print(sent.text)
    #writefile = open(f'./result/textrank/{i}.txt', 'w')
    writefile = open(f'./textrank_out_summ{i}.txt', 'w')
    writefile.write(result)
    writefile.close()
    break
    print(i)
print(datetime.now() - startTime)
