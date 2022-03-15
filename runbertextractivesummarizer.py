import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from summarizer import Summarizer
from summarizer.coreference_handler import CoreferenceHandler

# i = 1
for i in range (1, 4005):
    # read source documents
    reviewlist = []
    from datetime import datetime
    startTime = datetime.now()
    for dir2 in os.listdir(f'./data/topic_{i}/input_docs'):
        currfile = open(f'./data/topic_{i}/input_docs/{dir2}')
        temp = ''.join(currfile.readlines())
        temp = temp.replace('\n', ' ')
        temp = temp.replace('\t', ' ')
        reviewlist.append(temp)
        # print(temp)
    handler = CoreferenceHandler('en_core_web_md', greedyness=.4)
    model = Summarizer(sentence_handler=handler)
    result = ""
    for x in reviewlist:
        result = result + ''.join(model(x, min_length=10, ratio = 0.7))
    result2 = model(result, min_length=10, ratio = 0.4)
    full = ''.join(result2)
    writefile = open(f'./bertsum_out_summ{i}.txt', 'w')
    writefile.write(full)
    writefile.close()
    print(i)
    break
print(datetime.now() - startTime)

# for dir1 in os.listdir('./data/ICLR/2018'):
#     if os.path.exists(f'./data/ICLR/2018/{dir1}/meta_{dir1}.json'):
#         # with open(f'./data/ICLR/2018/{dir1}/meta_{dir1}.json') as f:
#         #     data = json.load(f)
#         #     ref = data["title"] + '. ' + data["comment"] + '. ' + data["decision"]
#         #     ref = ref.replace('\n', ' ')
#         #     ref = ref.replace('\t', ' ')
#         #     # os.mkdir(f'./multidoc_summarization/data/topic_{i}')
#         #     # os.mkdir(f'./multidoc_summarization/data/topic_{i}/references')
#         #     # os.mkdir(f'./multidoc_summarization/data/topic_{i}/input_docs')
#         #     refwrite = open(f'./multidoc_summarization/example_custom_dataset/topic_{i}_ref1.txt', 'w')
#         #     refwrite.write(ref)
#         #     refwrite.close()
#         # print(dir1[0:-5])
#         j = 1
#         reviewlist = []
#         for dir2 in os.listdir(f'./data/ICLR/2018/{dir1}'):
#             if dir2[0:4] == "meta" or dir2[0:4] == "comm":
#                 continue
            
#             with open(f'./data/ICLR/2018/{dir1}/{dir2}') as f:
#                 data = json.load(f)
#                 ref = data["title"] + '. ' + data["rating"][3:] + '. ' + data["review"] + '. ' + data["confidence"][3:]
#                 ref = ref.replace('\n', ' ')
#                 ref = ref.replace('\t', ' ')
#                 # refwrite = open(f'./multidoc_summarization/data/topic_{i}/input_docs/{j}.txt', 'w')
#                 # refwrite.write(ref)
#                 # refwrite.close()
#                 reviewlist.append(ref)
#             j = j + 1
#         handler = CoreferenceHandler('en_core_web_md', greedyness=.4)
#         model = Summarizer(sentence_handler=handler)
#         result = ""
#         for x in reviewlist:
#             result = result + ''.join(model(x, min_length=10, ratio = 0.7))
#         result2 = model(result, min_length=10, ratio = 0.4)
#         full = ''.join(result2)
#         print("=====" + dir1 + "=====")
#         print(full)
#         i = i + 1
