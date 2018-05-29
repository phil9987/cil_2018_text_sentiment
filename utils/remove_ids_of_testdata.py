import os
TEST_DATA = '../data/test_data.txt'                 # Path to test data (no labels, for submission)

def main():
    test = []
    for line in open(TEST_DATA, 'r', encoding='utf8'):
        split_idx = line.find(',')  # first occurrence of ',' is separator between id and tweet
        tweet = line[(split_idx + 1):]
        test.append(tweet)
    with open(TEST_DATA, 'w', encoding='utf8') as f:
        f.writelines(test)
    return 0


if __name__ == '__main__':
    main()