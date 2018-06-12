#!/usr/bin/env python3

import sys

def main():
    vecs = []
    for line in sys.stdin:
        smiles = line.count('<smile>')
        sadfaces = line.count('<sadface>')
        hearts = line.count('<heart>')
        neutralfaces = line.count('<neutralface>')
        lolfaces = line.count('<lolface>')
        users = line.count('<user>')
        hashtags = line.count('<hashtag>')
        elongs = line.count('<elong>')
        repeats = line.count('<repeat>')

        vecs.append([smiles, sadfaces, hearts, neutralfaces, lolfaces, users, hashtags, elongs, repeats])

    for el in vecs:
        print(' '.join(str(x) for x in el))
        
    return 0


if __name__ == '__main__':
    main()
