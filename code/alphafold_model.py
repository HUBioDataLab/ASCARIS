from collections import Counter
import glob
def reduce_model_dict(dict):
    for key, val in dict.items():
        used = []
        for key2, val2 in val.items():
            new = []
            for i in val2:
                if i not in used:
                    new.append(i)
                    used.append(i)
                val[key2] = new
    return dict


def which_model(position):
    models_dict = {}
    x = 1
    for i, j in zip(range(1400, 27000, 200), range(1, 27000, 200)):
        if position <= i and position >= j:
            models_dict[x] = position
        x += 1
    return models_dict

def modelCount(path_to_models):
    count_list = []
    for file in list(path_to_models.glob("*")):
        try:
            protein_id = str(file).split('-')[1]
            count_list.append(protein_id)
        except:
            IndexError
    count_dict = Counter(count_list)
    count_dict = {';'.join(sorted(k for k in count_dict.keys() if count_dict[k] == v)): v for v in
                  set(count_dict.values())}
    return count_dict