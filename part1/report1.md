# Isaiah Kriegman ישעיה קריגמן 348264771

## 1.3

> Provide a PDF file named report1.pdf, in which you provide a summary of the experiment: how large were the training and test sets, did your network succeed in distinguishing the two languages (it should)? how long did it take (both wall-clock time (i.e., number of seconds), and number of iterations)? did it succeed only on the train and not on the test? what you did in order to make it work, etc.

I generated 500 positive and negative exampels each and used an 80-20 train test split in the gen_examples.py file. It totally worked very quickly, getting 90-100% test set accuracy in 3-5 epochs with my current hyperparameters. Each epoch took less than a second on my 2019 MacBook intel CPU. 

At first, I skipped learning vector embeddings for each character; I just used `ord(c)` and checked if that was enough. It didn't work, and when I switched to having learned embeddings of dim 32, then it could actually work and got 100% test set accuracy. As an experiment, I took the embedding dim down to 2 and then 1 and found that it could also get >90% accuracy in 5 epochs. So it seems that 32 is overkill for this task and 1 works, just so long as it is learnable. For all these experiments, I kept the hidden dim at 32; bringing that down to 1 as well made the experiment fail.
