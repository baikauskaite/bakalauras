from german_nouns.lookup import Nouns
from pprint import pprint
import json

BASE_PATH = "/home/viktorija/bakalaurinis/log-probability-bias/tests/german/weat0.jsonl"
test = json.load(open(BASE_PATH, 'r', encoding='utf-8'))
attr1 = test['attr1']['examples']
attr2 = test['attr2']['examples']
new_attr1 = set()
new_attr2 = set()

nouns = Nouns()

dict = {
    'm': {
        'attr': attr1,
        'new_attr': new_attr1
    },
    'f': {
        'attr': attr2,
        'new_attr': new_attr2
    }
}

for gender, value in dict.items():
    for word in value['attr']:
        if nouns[word]:
            entry = nouns[word]
            noun = None

            for e in entry:
                pprint(e)
                if 'genus' in e.keys() and e['genus'] == gender:
                    noun = e
                    break

            if noun is not None:
                value['new_attr'].add(noun['lemma'])

print('m len:', len(new_attr1))
print('f len:', len(new_attr2))

# Make sets same length
if len(new_attr1) > len(new_attr2):
    new_attr1 = list(new_attr1)[:len(new_attr2)]
else:
    new_attr2 = list(new_attr2)[:len(new_attr1)]

print('new len:', len(new_attr1))

# Save the new attributes
new_test = test
new_test['attr1']['examples'] = list(new_attr1)
new_test['attr2']['examples'] = list(new_attr2)

with open(BASE_PATH.replace('.jsonl', '_new.jsonl'), 'w', encoding='utf-8') as f:
    json.dump(new_test, f, ensure_ascii=False, indent=4)

print('Done')