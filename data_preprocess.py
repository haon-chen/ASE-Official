import json
from tqdm import tqdm
import random
random.seed(0)

def make_test_data(fromfile, tofile, is_small=False):
    DOC_MAX_LEN = 20
    QUERY_MAX_LEN = 9
    with open(fromfile, "r") as fr:
        with open(tofile, "w") as fw:
            lines = fr.readlines()
            if is_small:
                lines = random.sample(lines, 200)
            for line in tqdm(lines):
                line = json.loads(line)
                history = ""
                for i in range(len(line["query"])):
                    q = line["query"][i]
                    sim_q = " ".join(q["candidates"][1].split()[:QUERY_MAX_LEN]) 
                    if i == len(line["query"])-1:
                        next_q = "[empty_q]"
                    else:
                        next_q = " ".join(line["query"][i+1]["text"].split()[:QUERY_MAX_LEN])
                    has_click = False
                    history += " ".join(q["text"].split()[:QUERY_MAX_LEN]) + "\t"
                    for doc in q["clicks"]:
                        if doc["label"] == True:
                            click_doc = " ".join(doc["title"].split()[:DOC_MAX_LEN])
                            break
                        else:
                            click_doc = "[empty_d]"
                    gen_labels = next_q + "\t" + click_doc + "\t" + sim_q
                    for doc in q["clicks"]:
                        title = " ".join(doc["title"].split()[:DOC_MAX_LEN])
                        if doc["label"] == True:
                            sample = "1" + "\t" + gen_labels + "\t" + history + title + "\n"
                            fw.write(sample)
                            if not has_click:
                                new_history = history + title + "\t"
                                has_click = True
                        else:
                            sample = "0" + "\t" + gen_labels + "\t" + history + title + "\n"
                            fw.write(sample)
                    history = new_history

def make_train_data(fromfile, tofile, is_small=False):
    DOC_MAX_LEN = 20
    QUERY_MAX_LEN = 9
    with open(fromfile, "r") as fr:
        with open(tofile, "w") as fw:
            lines = fr.readlines()
            if is_small:
                lines = random.sample(lines, 20000)
            for line in tqdm(lines):
                line = json.loads(line)
                history = ""
                true_hisory_dict = {}
                for i in range(len(line["query"])):
                    q = line["query"][i]
                    sim_q = " ".join(q["candidates"][0].split()[:QUERY_MAX_LEN]) 
                    next_d = "[empty_d]"
                    if i == 0:
                        previous_q1 = "[empty_q]"
                    else:
                        previous_q1 = " ".join(line["query"][i-1]["text"].split()[:QUERY_MAX_LEN])
                    if i <= 1:
                        previous_q2 = "[empty_q]"
                    else:
                        previous_q2 = " ".join(line["query"][i-2]["text"].split()[:QUERY_MAX_LEN])
                    
                    if i >= len(line["query"])-1:
                        next_q1 = "[empty_q]"
                    else:
                        next_q1 = " ".join(line["query"][i+1]["text"].split()[:QUERY_MAX_LEN])
                        for doc in line["query"][i+1]["clicks"]:
                            if doc["label"] == True:
                                next_d = " ".join(doc["title"].split()[:DOC_MAX_LEN])
                                break
                    if i >= len(line["query"])-2:
                        next_q2 = "[empty_q]"
                    else:
                        next_q2 = " ".join(line["query"][i+2]["text"].split()[:QUERY_MAX_LEN])
                    has_click = False
                    history += " ".join(q["text"].split()[:QUERY_MAX_LEN]) + "\t"
                    query_click_doc = []
                    query_unclick_doc = []
                    for doc in q["clicks"]:
                        title = " ".join(doc["title"].split()[:DOC_MAX_LEN])
                        if doc["label"] == True:
                            query_click_doc.append(history.strip() + "====" + title)
                            true_hisory_dict[history + title] = 1
                            if not has_click:
                                click_doc = " ".join(doc["title"].split()[:DOC_MAX_LEN])
                                new_history = history + title + "\t"
                                has_click = True
                        else:
                            query_unclick_doc.append(title)
                    assert len(query_click_doc) > 0
                    gen_labels = next_q1 + "\t" + next_q2 + "\t" + click_doc + "\t" + next_d  + "\t" + previous_q2 + "\t" + previous_q1 + "\t" + sim_q
                    for click in query_click_doc:
                        unclick_seq = ""
                        for unclick in query_unclick_doc:
                            # if click == unclick:
                            #     continue
                            unclick_seq += "\t" + unclick
                        unclick_cnt = len(query_unclick_doc)
                        fw.write("1" + "\t" + str(unclick_cnt)+ "\t" + gen_labels +  "\t" + click  + unclick_seq + "\n")
                    history = new_history

make_train_data("./data/aol/train_candidate.json", "./data/aol/train.txt")
make_test_data("./data/aol/test_candidate.json", "./data/aol/test.txt")
