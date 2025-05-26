import random

def random_digits(min_len=1, max_len=10):
    length = random.randint(min_len, max_len)
    return ''.join(random.choices('123456789', k=length))

def random_letters(letter, min_len=1, max_len=10):
    length = random.randint(min_len, max_len)
    return letter * length

def generate_positive_example():
    return (random_digits() + 
            random_letters('a') + 
            random_digits() + 
            random_letters('b') + 
            random_digits() + 
            random_letters('c') + 
            random_digits() + 
            random_letters('d') + 
            random_digits())

def generate_negative_example():
    return (random_digits() +
            random_letters('a') +
            random_digits() +
            random_letters('c') +  # c comes before b
            random_digits() +
            random_letters('b') +
            random_digits() +
            random_letters('d') +
            random_digits())

def write_examples(filename, examples):
    with open(filename, 'w') as f:
        for ex in examples:
            f.write(ex + '\n')

if __name__ == '__main__':
    num_pos = 500
    num_neg = 500

    pos_examples = [generate_positive_example() for _ in range(num_pos)]
    neg_examples = [generate_negative_example() for _ in range(num_neg)]

    write_examples('pos_examples', pos_examples)
    write_examples('neg_examples', neg_examples)

    # generate train/test sets
    pos_examples = [ex + " 1" for ex in pos_examples]
    neg_examples = [ex + " 0" for ex in neg_examples] 
    data = pos_examples + neg_examples
    random.shuffle(data)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    write_examples('train.txt', train_data)
    write_examples('test.txt', test_data)