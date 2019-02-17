def ThreeFiveWords(word):
    if len(word) % 3 == 0:
        if len(word) % 5 == 0:
            return 'three&five'
        return 'three'
    elif len(word) % 5 == 0:
        return 'five'
    return word
    
print(ThreeFiveWords(input()))