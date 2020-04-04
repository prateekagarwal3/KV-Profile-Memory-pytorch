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
print(model_personas[0])
print('~~~~~~~~~~~~~')
print(user_messages[0])
print('~~~~~~~~~~~~~')
print(model_response_targets[0])
print('~~~~~~~~~~~~~')
print(model_response_candidates[0])