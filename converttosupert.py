import os
import json

i = 3148
for dir1 in os.listdir('./data/ICLR/2019'):
    if os.path.exists(f"./data/ICLR/2019/{dir1}/meta_{dir1}.json"):
        with open(f'./data/ICLR/2019/{dir1}/meta_{dir1}.json') as f:
            data = json.load(f)
            ref = data["title"] + '. ' + data["comment"] + '. ' + data["decision"]
            ref = ref.replace('\n', ' ')
            ref = ref.replace('\t', ' ') 
            os.mkdir(f'./SUPERT/data/topic_{i}')
            os.mkdir(f'./SUPERT/data/topic_{i}/references')
            os.mkdir(f'./SUPERT/data/topic_{i}/input_docs')
            refwrite = open(f'./SUPERT/data/topic_{i}/references/ref1.txt', 'w')
            refwrite.write(ref)
            refwrite.close()
        # print(dir1[0:-5])
        j = 1
        for dir2 in os.listdir(f'./data/ICLR/2019/{dir1}'):
            if dir2[0:4] == "meta" or dir2[0:4] == "comm":
                continue
            with open(f'./data/ICLR/2019/{dir1}/{dir2}') as f:
                data = json.load(f)
                ref = data["title"] + '. ' + data["rating"][3:] + '. ' + data["review"] + '. ' + data["confidence"][3:]
                ref = ref.replace('\n', ' ')
                ref = ref.replace('\t', ' ')
                refwrite = open(f'./SUPERT/data/topic_{i}/input_docs/{j}.txt', 'w')
                refwrite.write(ref)
                refwrite.close()
            j = j + 1
        i = i + 1
