from pylexique import Lexique383
from pprint import pprint
import random

MALE_PATH = "/home/viktorija/bakalaurinis/context-debias/data/french/male.txt"
FEMALE_PATH = "/home/viktorija/bakalaurinis/context-debias/data/french/female.txt"

LEXIQUE = Lexique383()  # Load the lexique                      
m_nouns = []
mm_nouns = []
f_nouns = []
ff_nouns = []

with open(MALE_PATH, 'r', encoding='utf-8') as f:
    male_words = [word.strip() for word in f.readlines()]

with open(FEMALE_PATH, 'r', encoding='utf-8') as f:
    female_words = [word.strip() for word in f.readlines()]

for m_word, f_word in zip(male_words, female_words):
    if m_word in LEXIQUE.lexique and f_word in LEXIQUE.lexique:
        print(m_word, f_word)
        m_entry = LEXIQUE.get_lex(m_word)
        f_entry = LEXIQUE.get_lex(f_word)

        if not isinstance(m_entry, list):
            m_entry = [m_entry]
        if not isinstance(f_entry, list):
            f_entry = [f_entry]

        m_flag = False
        mm_flag = False
        f_flag = False
        ff_flag = False

        for e in m_entry:
            if e.cgram == 'NOM' and e.ortho not in m_nouns:
                m_nouns.append(e.ortho)
                m_flag = True
                if e.genre == 'm' and e.ortho not in mm_nouns:
                    mm_nouns.append(e.ortho)
                    mm_flag = True
                    break

        for e in f_entry:
            if e.cgram == 'NOM' and m_flag and e.ortho not in f_nouns:
                f_nouns.append(e.ortho)
                f_flag = True
                if e.genre == 'f' and mm_flag and e.ortho not in ff_nouns:
                    ff_nouns.append(e.ortho)
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

# Save the new stereotypes
with open(MALE_PATH.replace('.txt', '_new.txt'), 'w', encoding='utf-8') as f:
    for word in m_nouns:
        f.write(word + '\n')

with open(FEMALE_PATH.replace('.txt', '_new.txt'), 'w', encoding='utf-8') as f:
    for word in f_nouns:
        f.write(word + '\n')