import csv, pickle
from geopy.distance import geodesic
from gensim import corpora, models
from gensim.test.utils import datapath

'''
newport_ri = (41.49008, -71.312796)
cleveland_oh = (41.499498, -71.695391)
print(geodesic(newport_ri, cleveland_oh).m)
'''

temp_file = datapath("C:/Users/xuteng/PycharmProjects/MDM20/LDA_model/lda_trained_model")
lda_model = models.LdaModel.load(temp_file)

for i in range(lda_model.num_topics):
    print(lda_model.print_topics()[i])

'''
with open('LDA_model/train_cleaned_text.csv', 'r') as handle:
    rf = csv.reader(handle, delimiter='|')

    with open('LDA_model/train_cleaned_text_2.csv', 'a', newline='') as whandle:
        wf = csv.writer(whandle, delimiter='|')

        for each_row in rf:
            if each_row:
                wf.writerow([each_row[0], each_row[1]])
'''