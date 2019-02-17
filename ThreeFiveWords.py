def ThreeFiveWords(word):
    if len(word) % 3 == 0 and len(word) % 5 == 0:
        return 'three&five'
    elif len(word) % 5 == 0:
        return 'five'
    elif len(word) % 3 == 0:
        return 'three'
    else:
        return word
    
print(ThreeFiveWords(input('Type a word: ')))