import os
TRAINING_DATA_POS = '../data/train_pos.txt'    # Path to positive training data
TRAINING_DATA_NEG = '../data/train_neg.txt'    # Path to negative training data

def main():
    test = []
    for line in open(TRAINING_DATA_POS, 'r', encoding='utf8'):
        line = line.replace(') ) )', '<russian_smile3>') 
        line = line.replace(')))', '<russian_smile3>') 
        line = line.replace(') )', '<russian_smile2>')
        line = line.replace('))', '<russian_smile2>')
        line = line.replace(')', '<russian_smile>')
        line = line.replace('( ( (', '<russian_sadsmile3>')
        line = line.replace('(((', '<russian_sadsmile3>')
        line = line.replace('( (', '<russian_sadsmile2>')
        line = line.replace('((', '<russian_sadsmile2>')
        line = line.replace('(', '<russian_sadsmile2>')
        russian_smile3_count = line.count('<russian_smile3>')
        russian_smile2_count = line.count('<russian_smile2>')
        russian_smile_count = line.count('<russian_smile>')
        russian_sadsmile3_count = line.count('<russian_sadsmile3>')
        russian_sadsmile2_count = line.count('<russian_sadsmile2>')
        russian_sadsmile_count = line.count('<russian_sadsmile>')
        test.append("{},{},{},{},{},{}".format(russian_smile3_count, russian_smile2_count, russian_smile_count, russian_sadsmile3_count, russian_sadsmile2_count, russian_sadsmile_count))
    with open('emoji_vectors.txt', 'w', encoding='utf8') as f:
        for el in test:
            f.write(el + '\n')
    return 0


if __name__ == '__main__':
    main()