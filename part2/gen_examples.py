import random
import argparse
import os

def random_digits(min_len=1, max_len=10):
    length = random.randint(min_len, max_len)
    return ''.join(random.choices('123456789', k=length))

def random_letters(letter, min_len=1, max_len=10):
    length = random.randint(min_len, max_len)
    return letter * length

def generate_balanced_brackets(depth=0, max_depth=20):
    brackets = [('{', '}'), ('[', ']'), ('(', ')')]
    if depth >= max_depth:
        return ''
    
    bracket_type = random.choice(brackets)
    left, right = bracket_type
    inner = generate_balanced_brackets(depth + 1, max_depth)
    return left + inner + right

def generate_imbalanced_brackets():
    brackets = ['{', '}', '[', ']', '(', ')']
    length = random.randint(30, 50)
    return ''.join(random.choices(brackets, k=length))

def generate_original():
    return (random_digits() + 
            random_letters('a') + 
            random_digits() + 
            random_letters('b') + 
            random_digits() + 
            random_letters('c') + 
            random_digits() + 
            random_letters('d') + 
            random_digits())

def generate_even_start_positive():
    first_digit = random.choice('02468')
    return first_digit + generate_original()[1:]

def generate_even_start_negative():
    first_digit = random.choice('13579')
    return first_digit + generate_original()[1:]

def generate_same_ends_positive():
    example = generate_original()
    return example[0] + example[1:] + example[0]

def generate_same_ends_negative():
    example = generate_original()
    while example[0] == example[-1]:
        example = generate_original()
    return example

def generate_pal(length=10):
    example = ''
    while len(example) < 2*length:
        example = generate_original()
    prefix = example[:length]
    example = prefix + prefix[::-1] + example[length:]
    return example

def generate_pal_neg(length=10):
    example = ''
    while len(example) < 2*length:
        example = generate_original()
    while len(example) < 2*length and example[:length] == example[length:length*2]:
        example = generate_original()
    return example    

def write_examples(filename, examples):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for ex in examples:
            f.write(ex + '\n')

def generate_and_save(language: str, num_pos: int, num_neg: int):
    if language == "brackets":
        pos_gen = generate_balanced_brackets
        neg_gen = generate_imbalanced_brackets
        dir_name = "brackets"
    elif language == "even_start":
        pos_gen = generate_even_start_positive
        neg_gen = generate_even_start_negative
        dir_name = "even_start"
    elif language == "same_ends":
        pos_gen = generate_same_ends_positive
        neg_gen = generate_same_ends_negative
        dir_name = "same_ends"
    elif language == "palindrome":
        pos_gen = generate_pal
        neg_gen = generate_pal_neg
        dir_name = "palindrome"
    else:
        raise ValueError(f"Unknown language: {language}")

    pos_examples = [pos_gen() for _ in range(num_pos)]
    neg_examples = [neg_gen() for _ in range(num_neg)]

    # Write raw examples
    write_examples(f"{dir_name}/pos_examples", pos_examples)
    write_examples(f"{dir_name}/neg_examples", neg_examples)

    # Generate train/test sets
    pos_examples = [ex + " 1" for ex in pos_examples]
    neg_examples = [ex + " 0" for ex in neg_examples]
    data = pos_examples + neg_examples
    random.shuffle(data)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    write_examples(f"{dir_name}/train.txt", train_data)
    write_examples(f"{dir_name}/test.txt", test_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, required=True,
                        choices=['brackets', 'even_start', 'same_ends', 'palindrome'],
                        help='Language to generate examples for')
    parser.add_argument('--num_pos', type=int, default=500,
                        help='Number of positive examples (default: %(default)d)')
    parser.add_argument('--num_neg', type=int, default=500,
                        help='Number of negative examples (default: %(default)d)')
    
    args = parser.parse_args()
    generate_and_save(args.language, args.num_pos, args.num_neg)