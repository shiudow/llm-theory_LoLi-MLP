# llm-theory_LoLi-MLP
"LogLinearActiveHead (LiLi-MLP)" is a learning model based on the hypothesis that "integer-scale and logarithmic-scale patterns are embedded within a single context."

[日本語版](./README_ja.md)

# The Importance of Selective Non-Learning: Realizing Zero-Activation Sparse Transformers via Log-Linear Geometrical Clamping)

(This is tranlated by translate.google.co.jp)

Given the current semiconductor shortage, I hope this theory will lead to improvements in the computational efficiency of LLMs.

## Abstract
"Learning unnecessary noise" is leading actually reducing learning ability, In the modern approach of large-scale language model (LLM) development,
This paper proposes the hypothesis that "thoroughly ignoring irrelevant noise and avoiding indiscriminate learning" is crucial for efficient intelligence growth.
As a geometrical empirical structure, I propose "LogLinearActiveHead (LoLi-MLP)" as one of the models that can be learned in a short time.

"LiLi-MLP" is a learning model based on the hypothesis that "integer-scale and logarithmic-scale patterns are embedded within a single context."

It achieves this by "fixing the representation of the geometric relationships of semantic vectors to only integer-scale and logarithmic-scale data," and "not learning outliers as invalid data."

The implementation involved replacing only the head layer of "Qwen/Qwen2.5-0.5B."
In other words, the tokenizer and attention were used as they were in Qwen2.5, and only the context relationships were trained using Japanese Wikipedia data.

A single NVidia RX5070 was used for model training.
 After training with Japanese Wikipedia data, a Loss of 2.0982 was achieved in about 5 hours.

Subsequently, as a final adjustment to the Japanese response, it was trained on Japanese conversation data for about 5 minutes and fine-tuned.
While the responses were still slightly off, they were sufficient considering the amount of training time.

## How to Use

* [How to Run the Sample][(./usage.md)]

## LogLinearActiveHead (LoLi-MLP)

This model works within the LLM processing flow of "input -> attention -> rotation (orthogonalization) -> head -> output" by employing two mechanisms:

1. Instead of inputting a semantic vector x to the head, it uses a vector directly connecting "x" and "log x" as input.
2. To prevent outlier learning, during quasi-negotiation, values ​​less than or equal to 0 are treated as 0, values ​​greater than or equal to 1 are treated as 1, and during backpropagation, the gradient is set to 0 to prevent learning.

This is achieved through these two mechanisms.

## What is the use of this theory?


1. Accelerating Model Training Time
    a. The more data there is, the faster the training becomes.
    b. The more data there is, the deeper the lower bound of the Loss becomes.
2. Zero-Sparse Processing
    a. Power consumption can be reduced by using GPU functionality to accelerate zero-operation processing.
    b. A model that is easy to use for learning.

Even these two points alone are considered to have a significant cost-saving effect for companies that create and operate commercial LLMs.

## Challenges

1. The Need for Evaluation Metrics for Model Training Status Other Than Loss
    a. With training of about 10 minutes, while the grammatical structure of Japanese can be learned, the accuracy of predicting detailed word candidates cannot be learned.
    b. Nevertheless, the Loss value is often 5.0 or less.
2. Zero-Sparse Processing of the Model Itself
    a. At first, this model is considered to build 0 sparse LLM model
    b. My GPU card has only 12GB MEM. So, I want the 0-sparse LLM model for use it in local PC.
3. LoRA adapter
    a. This model is easy to use for learning in local PC.

## License

Code: Apache License 2.0
Model Weights: CC BY-NC-SA 4.0 (Derived from msfm/ichikara-instruction-all)
