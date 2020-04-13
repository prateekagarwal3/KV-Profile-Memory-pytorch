# path = 'data/personachat/train_self_original.txt'
path = 'data/example_data.txt'
user_messages = []
model_response_targets = []
model_response_candidates = []
model_personas = []
cnt = 0
cands_are_replies = False
i = 0

def parse_persona(profile):
    modified_profile = []
    for p in profile:
        p = p[13:]
        modified_profile.append(p)
    return modified_profile

with open(path) as read:
    for line in read:
        line = line.strip().replace('\\n', '\n')
        if len(line) > 0:
            cnt = cnt + 1
            if cnt == 1 and line[0:2] == '1 ':
                lines_have_ids = True
            if '\t' in line and not cands_are_replies:
                cands_are_replies = True
                model_personas.append(parse_persona(model_response_targets))
                model_response_targets = []
            if lines_have_ids:
                space_idx = line.find(' ')
                line = line[space_idx + 1 :]
                if cands_are_replies:
                    sp = line.split('\t')
                    if len(sp) > 1 and sp[1] != '':
                        user_messages.append(sp[0])
                        model_response_targets.append(sp[1])
                        model_response_candidates.append(sp[3])
                    i +=1
                else:
                    model_response_targets.append(line)
            else:
                model_response_targets.append(line)

# for i in range(len(model_response_candidates)):
#     model_response_candidates[i] = model_response_candidates[i].split('|')

print(len(model_personas))
print('~~~~~~~~~~~~~')
print(user_messages)
print('~~~~~~~~~~~~~')
print(len(model_response_targets))
print('~~~~~~~~~~~~~')
print(len(model_response_candidates))

# train_data = []
# training_example = {}
# for i in range(len(user_messages)):
#     training_example["user_message"] = user_messages[i]
#     training_example["model_response_candidates"] = model_response_candidates[i]
#     training_example["model_response_target"] = model_response_targets[i]
#     training_example["model_persona"] = model_personas[0]
#     train_data.append(training_example)
# print('no of training examples are {}'.format(len(train_data)))