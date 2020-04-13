path = 'data/personachat/train_self_original.txt'
# path = 'data/example_data.txt'
user_messages = []
model_response_targets = []
model_response_candidates = []
model_persona = None
train_data = []
cands_are_replies = False

def parse_persona(profile):
    modified_profile = []
    for p in profile:
        p = p[13:]
        modified_profile.append(p)
    return modified_profile

def split_model_response_candidates(model_response_candidates):
    model_response_candidates = model_response_candidates.split('|')
    return model_response_candidates

with open(path) as read:
    for line in read:
        line = line.strip().replace('\\n', '\n')
        if len(line) > 0:
            if line[0:2] == '1 ':
                lines_have_ids = True
                if len(user_messages) != 0:
                    for i in range(len(user_messages)):
                        training_example = {}
                        training_example["user_message"] = user_messages[i]
                        training_example["model_response_candidates"] = model_response_candidates[i]
                        training_example["model_response_target"] = model_response_targets[i]
                        training_example["model_persona"] = model_persona
                        train_data.append(training_example)
                    user_messages = []
                    model_response_targets = []
                    model_response_candidates = []
                    model_persona = None
                    cands_are_replies = False
            if '\t' in line and not cands_are_replies:
                cands_are_replies = True
                model_persona = parse_persona(model_response_targets)
                model_response_targets = []
            if lines_have_ids:
                space_idx = line.find(' ')
                line = line[space_idx + 1 :]
                if cands_are_replies:
                    sp = line.split('\t')
                    if len(sp) > 1 and sp[1] != '':
                        user_messages.append(sp[0])
                        model_response_targets.append(sp[1])
                        model_response_candidates.append(split_model_response_candidates(sp[3]))
                else:
                    model_response_targets.append(line)
            else:
                model_response_targets.append(line)
    for i in range(len(user_messages)):
        training_example = {}
        # print(user_messages[i])
        training_example["user_message"] = user_messages[i]
        training_example["model_response_candidates"] = model_response_candidates[i]
        training_example["model_response_target"] = model_response_targets[i]
        training_example["model_persona"] = model_persona
        train_data.append(training_example)

# print('~~~~~~~~~~~~~')
print('no of training examples are {}'.format(len(train_data)))