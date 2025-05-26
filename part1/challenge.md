# Isaiah Kriegman ישעיה קריגמן 348264771

## 1.1

* The two languages cannot be distinguished using a bag of word approach, they are the same bag of words. Only the order is different. 

* It's sort of hard for me to picture what a  bigram/trigram based approach would look like. I suppose that it could if there was a fixed total length and the model learned to find the letters in certain positions. It would depend on the distribution of "+" in the regex. Like if it was always 2 then it would work. But I assume you want it to be something like uniform[10] in which case it wouldn't work, because the consequential information would show up at conflicting fixed positions. 

* I feel like a CNN approach could work but it would be hard? Like a CNN "boils" the information down locally, like blurring a picture. And blurring a picture is a analagous to how I would approach this problem. I could see a CNN learning to filter out the numbers, but keep the letters, and basically boil down a fixed sequence length down to a vector of a few components representing the order of the letters in the sequence. We would need enough CNN layers to boil down the max length to something small enough. I suppose that if "+" were interpreted to mean up to 1,000, then it might be challening for the network to learn. We might need 1,000 CNN layers.