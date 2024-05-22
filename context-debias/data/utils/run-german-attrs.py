from german_nouns.lookup import Nouns
from pprint import pprint
import random

MALE_PATH = "/home/viktorija/bakalaurinis/context-debias/data/german/male.txt"
FEMALE_PATH = "/home/viktorija/bakalaurinis/context-debias/data/german/female.txt"

nouns = Nouns()               
m_nouns = []
mm_nouns = []
f_nouns = []
ff_nouns = []

with open(MALE_PATH, 'r', encoding='utf-8') as f:
    male_words = [word.strip() for word in f.readlines()]

with open(FEMALE_PATH, 'r', encoding='utf-8') as f:
    female_words = [word.strip() for word in f.readlines()]

for m_word, f_word in zip(male_words, female_words):
    if nouns[m_word] and nouns[f_word]:
        print(m_word, f_word)
        m_entry = nouns[m_word]
        f_entry = nouns[f_word]

        m_flag = False
        mm_flag = False
        f_flag = False
        ff_flag = False

        for e in m_entry:
            if m_word not in m_nouns:
                m_nouns.append(m_word)
                m_flag = True
                if 'genus' in e.keys() and e['genus'] == 'm' and m_word not in mm_nouns:
                    mm_nouns.append(m_word)
                    mm_flag = True
                    break

        for e in f_entry:
            if m_flag and f_word not in f_nouns:
                f_nouns.append(f_word)
                f_flag = True
                if 'genus' in e.keys() and e['genus'] == 'f' and mm_flag and f_word not in ff_nouns:
                    ff_nouns.append(f_word)
                    ff_flag = True
                    break
        
        if m_flag and not f_flag:
            m_nouns.pop()

        if mm_flag and not ff_flag:
            mm_nouns.pop()

print('m len:', len(m_nouns))
print('f len:', len(f_nouns))
print('mm len:', len(mm_nouns))
print('ff len:', len(ff_nouns))

# # Save the new stereotypes
with open(MALE_PATH.replace('.txt', '_new.txt'), 'w', encoding='utf-8') as f:
    for word in m_nouns:
        f.write(word + '\n')

with open(FEMALE_PATH.replace('.txt', '_new.txt'), 'w', encoding='utf-8') as f:
    for word in f_nouns:
        f.write(word + '\n')