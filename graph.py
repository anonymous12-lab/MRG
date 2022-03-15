from typing import Sized
import json
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os

src_path='/content/drive/MyDrive/ML/innovation_lab/aspect_normal/aspect'
task1_path=os.path.join(src_path,'task1')
task2_path=os.path.join(src_path,'task2')

labels=['Bert-Extractive-Summarizer','Pegasus','SUPERT','Textrank']

data_lst=[]
for json_file in sorted(os.listdir(task1_path)):
  if json_file.split('_')[0]!='output':
    continue
  print(json_file)
  json_l=[]
  with open(os.path.join(task1_path,json_file),'r') as f:
    for i in f:
      json_l.append(json.loads(i))

  dict_1={0:'MET',1:'EXP',2:'RWK',3:'RES',4:'PDI',5:'OAL',6:'DAT',7:'INT',8:'ANA',9:'TNF',10:'BIB',11:'EXT',12:'CNT',13:'FWK',14:'ABS'}
  counter_1={'MET':0,'EXP':0,'RWK':0,'RES':0,'PDI':0,'OAL':0,'DAT':0,'INT':0,'ANA':0,'TNF':0,'BIB':0,'EXT':0,'CNT':0,'FWK':0,'ABS':0}
  ori_1={'ABS':0,'INT':0,'RWK':0,'PDI':0,'DAT':0,'MET':0,'EXP':0,'RES':0,'TNF':0,'ANA':0,'FWK':0,'OAL':0,'BIB':0,'EXT':0}

  for each_meta_review in json_l:

    action_logits=each_meta_review['action_logits']
    mr_labels=[]
    for sent in action_logits:
      argmax_index=np.argmax(np.array(sent),axis=0).tolist()
      # print(argmax_index)
      # argmax_labels=[dict_[str(p)] for p in argmax_index]
      mr_labels.append(dict_1[argmax_index])
    for l in mr_labels:
      counter_1[l]=counter_1[l]+1
    # pdb.set_trace()
    
  for key in ori_1:
    ori_1[key]=counter_1[key]
  data_cat=[]
  for key in ori_1:
    data_cat.append(ori_1[key])
  data_lst.append(data_cat)
  print(counter_1)

data_keys = ori_1.keys()
X = np.arange(len(ori_1))
fig, ax = plt.subplots(figsize=(16,5))
# fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, data_lst[0], color = '#fc2924', width = 0.20)
ax.bar(X + 0.20, data_lst[1], color = '#fce524', width = 0.20)
ax.bar(X + 0.40, data_lst[2], color = '#04d924', width = 0.20)
ax.bar(X + 0.60, data_lst[3], color = '#4d24fc', width = 0.20)
ax.legend(labels)
ax.set_ylabel('frequency',size='15',fontweight='bold')
# ax.set_xlabel('section',size='15',fontweight='bold')
ax.set_xticks(X+0.30)
ax.set_xticklabels(data_keys,fontweight='bold')
plt.xticks(rotation=90)
# pps = [ax.bar(X + 0.00, data_lst[0], color = '#fc2924', width = 0.20),
# ax.bar(X + 0.20, data_lst[1], color = '#fce524', width = 0.20),
# ax.bar(X + 0.40, data_lst[2], color = '#04d924', width = 0.20),
# ax.bar(X + 0.60, data_lst[3], color = '#4d24fc', width = 0.20)]
# for item in pps:
#   for p in item:
#     height = p.get_height()
#     ax.annotate('{}'.format(height),
#         xy=(p.get_x() + p.get_width() / 2, height),
#         xytext=(0, 3), # 3 points vertical offset
#         textcoords="offset points",
#         ha='center', va='bottom')
fig.savefig("/content/drive/MyDrive/ML/innovation_lab/aspect_normal/task1", bbox_inches='tight',pad_inches=0.1)
plt.show()
plt.clf()

print("\n")

data_lst=[]
for json_file in sorted(os.listdir(task2_path)):
  if json_file.split('_')[0]!='output':
    continue
  print(json_file)
  json_l=[]
  with open(os.path.join(task2_path,json_file),'r') as f:
    for i in f:
      json_l.append(json.loads(i))

  dict_2={0:'EMP',1:'NULL',2:'CMP',3:'SUB',4:'CLA',5:'PNF',6:'IMP',7:'NOV',8:'REC',9:'CNT',10:'APR'}
  counter_2={'EMP':0,'NULL':0,'CMP':0,'SUB':0,'CLA':0,'PNF':0,'IMP':0,'NOV':0,'REC':0,'CNT':0,'APR':0}
  ori_2={'APR':0,'NOV':0,'IMP':0,'CMP':0,'PNF':0,'REC':0,'EMP':0,'SUB':0,'CLA':0}

  for each_meta_review in json_l:

    action_logits=each_meta_review['action_logits']
    mr_labels=[]
    for sent in action_logits:
      argmax_index=np.argmax(np.array(sent),axis=0).tolist()
      # print(argmax_index)
      # argmax_labels=[dict_[str(p)] for p in argmax_index]
      mr_labels.append(dict_2[argmax_index])
    for l in mr_labels:
      counter_2[l]=counter_2[l]+1
    # pdb.set_trace()

  for key in ori_2:
    ori_2[key]=counter_2[key]
  data_cat=[]
  for key in ori_2:
    data_cat.append(ori_2[key])
  data_lst.append(data_cat)
  print(counter_2)

data_keys = ori_2.keys()
X = np.arange(len(ori_2))
fig, ax = plt.subplots(figsize=(11,5))
# fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, data_lst[0], color = '#fc2924', width = 0.20)
ax.bar(X + 0.20, data_lst[1], color = '#fce524', width = 0.20)
ax.bar(X + 0.40, data_lst[2], color = '#04d924', width = 0.20)
ax.bar(X + 0.60, data_lst[3], color = '#4d24fc', width = 0.20)
ax.legend(labels)
ax.set_ylabel('frequency',size='15',fontweight='bold')
# ax.set_xlabel('aspect',size='15',fontweight='bold')
ax.set_xticks(X+0.30)
ax.set_xticklabels(data_keys,fontweight='bold')
# pps = [ax.bar(X + 0.00, data_lst[0], color = '#fc2924', width = 0.20),
# ax.bar(X + 0.20, data_lst[1], color = '#fce524', width = 0.20),
# ax.bar(X + 0.40, data_lst[2], color = '#04d924', width = 0.20),
# ax.bar(X + 0.60, data_lst[3], color = '#4d24fc', width = 0.20)]
# for item in pps:
#   for p in item:
#     height = p.get_height()
#     ax.annotate('{}'.format(height),
#         xy=(p.get_x() + p.get_width() / 2, height),
#         xytext=(0, 3), # 3 points vertical offset
#         textcoords="offset points",
#         ha='center', va='bottom')
fig.savefig("/content/drive/MyDrive/ML/innovation_lab/aspect_normal/task2", bbox_inches='tight',pad_inches=0.1)
plt.show()
plt.clf()