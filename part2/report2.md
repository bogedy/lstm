# Isaiah Kriegman ישעיה קריגמן 348264771

## 2

> * A description of the languages.
> * Why did you think the language will be hard to distinguish?
> * Did you manage to fail the LSTM acceptor? (including, train and test set sizes, how many iterations did you train for, did it manage to learn the train but did not generalize well to the test, or did it fail also on train?)

Just like with part 1, I generated 500 positive and negative exampels each and used an 80-20 train test split in the gen_examples.py file. My criterion for "struggle" was just that it failed to show signs of meaningful learning under the same hyperparamters and dataset size that I used before. 

My three languages were the following:

* **Even start**: The same regex pattern as the positive examples in part 1, except that positive examples start with an even digit and negative examples start with an odd digit. The challenge was to learn to ignore everything after the first character. The train loss bounced around and didn't decrease, test accuracy hovered around 50%.

* **Same ends**: Again I took the same pattern from part 1. Positive examples had the same first and last character and negative did not. The results were the same. The challenge here was to learn to ignore everything in the middle. 

* **Palindrome** Positive examples start with a 20 character palindrome and negative were just the same regex pattern as in part 1. To generate the palindromes I reversed the first 10 characters in the generated string and inserted them at position 10. The results were the same as above. Interestingly, I also tried something similar to this where the *end* of the string was the beginning reversed. The LSTM actually learned this quite easily like in part 1. I am not sure why each of these would fail and succeed respectively, they feel similar. I guess it must come down to the palindrome language requiring ignoring the end of the string. 

I also tried a language of brackets {}[]() where positive sequences were balanced and negative sequences were not. The model learned this easily. I thought it would be too "logical" for the model to learn through LSTM gates, but I guess not! I think maybe a language with a balanced number of brackets but with bad syntax might trip up the model, but I didn't try that. 