import os
import pandas as pd
import tensorflow as tf

save_path = "./pegasus/pegasus/data/testdata/custom_test.tfrecord"

input_dict = dict(
                  inputs=[
                          # Your text inputs to be summarized.
                         ],
                  targets=[
                          # Corresponding targets for the inputs.
                          ]
                 )

for i in range (3299+706, 4713):
    # read source documents
    reviewlist = []
    for dir2 in os.listdir(f'./data/topic_{i}/input_docs'):
        currfile = open(f'./data/topic_{i}/input_docs/{dir2}')
        temp = ''.join(currfile.readlines())
        temp = temp.replace('\n', ' ')
        temp = temp.replace('\t', ' ')
        reviewlist.append(temp)
        # print(temp)
    review = '. '.join(reviewlist)
    input_dict["inputs"].append(review)
    if os.path.exists(f'./data/topic_{i}/references/ref1.txt'):
        currfile2 = open(f'./data/topic_{i}/references/ref1.txt')
        temp2 = ''.join(currfile2.readlines())
        temp2 = temp2.replace('\n', ' ')
        temp2 = temp2.replace('\t', ' ')
        input_dict["targets"].append(temp2)
    else:
        input_dict["targets"].append("")

# print(input_dict)
data = pd.DataFrame(input_dict)

with tf.io.TFRecordWriter(save_path) as writer:
    for row in data.values:
        inputs, targets = row[:-1], row[-1]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "inputs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[inputs[0].encode('utf-8')])),
                    "targets": tf.train.Feature(bytes_list=tf.train.BytesList(value=[targets.encode('utf-8')])),
                }
            )
        )
        writer.write(example.SerializeToString())