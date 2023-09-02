import os
source_dir = "../filelists"
prosody_file = os.path.join(source_dir, "000001-010000.txt")

def remove_punc(s):
    punc = '~`!$%^&*()_+-=|\';":/.,?><~·！@￥%……&*（）——+-=“：’；、。，？》《{}'
    res = ""
    for char in s:
        if char in punc:
            continue
        res = res + char
    return res
path_sent = []
with open(prosody_file,"r") as log:
    lines = log.readlines()
    idx = 0
    while idx < len(lines):
        sentence = lines[idx].strip().split()[1]
        filename = "dummy/" + lines[idx].strip().split()[0] + ".wav"

        sentence_remove = sentence.strip().replace("#1","").replace("#2","").replace("#3","").replace("#4","")
        sentence_remove = remove_punc(sentence_remove)
        piny = lines[idx+1].strip()
        if len(piny.split()) != len(sentence_remove):
            idx = idx + 2
            continue
        sentence_remove_punc = remove_punc(sentence)
        piny_split = piny.split()
        assert len(sentence_remove_punc) - 2*sentence_remove_punc.count("#") == len(piny_split)
        idx = idx + 2
        char_idx = 0
        piny_idx = 0
        piny_add_prosody = []
        while char_idx < len(sentence_remove_punc):
            if sentence_remove_punc[char_idx] == "#":
                piny_add_prosody.append(sentence_remove_punc[char_idx:char_idx+2])
                char_idx = char_idx + 2
            else:
                piny_add_prosody.append(piny_split[piny_idx])
                char_idx = char_idx + 1
                piny_idx = piny_idx + 1
        path_sent.append((filename, piny_add_prosody))

with open(os.path.join(source_dir,"csmsc_path_sent.txt"), "w") as log:
    for params in path_sent:
        path, sent = params
        sent = " ".join(sent)
        log.write(path + "|" + sent + "\n")
