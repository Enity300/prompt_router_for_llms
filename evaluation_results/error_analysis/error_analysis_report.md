# Comprehensive Error Analysis Report

**Generated:** 2025-11-17 08:38:04

## Overview

- **Models Analyzed:** 10
- **Total Comparisons:** 10

## Error Rate Summary

| Model | Total Samples | Errors | Error Rate | Accuracy |
|-------|---------------|--------|------------|----------|
| Sentence Transformer + CatBoost Router | 2902 | 62 | 0.0214 | 0.9786 |
| TF-IDF + Random Forest | 2902 | 108 | 0.0372 | 0.9628 |
| TF-IDF + SVM | 2902 | 120 | 0.0414 | 0.9586 |
| TF-IDF + Logistic Regression | 2902 | 122 | 0.0420 | 0.9580 |
| TF-IDF + CatBoost | 2902 | 204 | 0.0703 | 0.9297 |
| TF-IDF + Naive Bayes | 2902 | 449 | 0.1547 | 0.8453 |
| Rule-based Keywords | 2902 | 667 | 0.2298 | 0.7702 |
| Semantic Router | 2902 | 979 | 0.3374 | 0.6626 |
| Most Frequent Class | 2902 | 1719 | 0.5924 | 0.4076 |
| Random Classifier | 2902 | 1958 | 0.6747 | 0.3253 |

## Detailed Model Analysis

### Sentence Transformer + CatBoost Router

**Error Statistics:**
- Total Errors: 62
- Error Rate: 2.14%

**Most Common Confusion Patterns:**

- math → general_knowledge: 20 occurrences
- coding → general_knowledge: 16 occurrences
- general_knowledge → math: 13 occurrences
- math → coding: 8 occurrences
- general_knowledge → coding: 3 occurrences

**Error Distribution by Category:**

| Category | Total | Errors | Error Rate |
|----------|-------|--------|------------|
| Coding | 1183 | 18 | 1.52% |
| General Knowledge | 1100 | 16 | 1.45% |
| Math | 619 | 28 | 4.52% |

**Example Failure Cases:**

1. **Query:** "What is the smallest pair of twin primes greater than 40?"
   - **True Category:** math
   - **Predicted:** general_knowledge

2. **Query:** "What is the square root of 0.25?"
   - **True Category:** general_knowledge
   - **Predicted:** math

3. **Query:** "$2^{\sqrt{n}}>n^2$ inequality proof I am trying to do some trick with lim but get stuck on $2^{\sqrt{n}}>n^2$. I want to prove it for n large enough b..."
   - **True Category:** math
   - **Predicted:** general_knowledge

4. **Query:** "How to handle cookies in Express.js?"
   - **True Category:** coding
   - **Predicted:** general_knowledge

5. **Query:** "In darts, what is the highest possible checkout with three darts, finishing with a double?"
   - **True Category:** general_knowledge
   - **Predicted:** math

### TF-IDF + Random Forest

**Error Statistics:**
- Total Errors: 108
- Error Rate: 3.72%

**Most Common Confusion Patterns:**

- math → general_knowledge: 64 occurrences
- general_knowledge → math: 28 occurrences
- coding → general_knowledge: 7 occurrences
- general_knowledge → coding: 4 occurrences
- coding → math: 4 occurrences

**Error Distribution by Category:**

| Category | Total | Errors | Error Rate |
|----------|-------|--------|------------|
| Coding | 1183 | 11 | 0.93% |
| General Knowledge | 1100 | 32 | 2.91% |
| Math | 619 | 65 | 10.50% |

**Example Failure Cases:**

1. **Query:** "What is the smallest pair of twin primes greater than 40?"
   - **True Category:** math
   - **Predicted:** general_knowledge

2. **Query:** "Steve is 5'6".  He grows 6 inches.  How tall is he in inches?"
   - **True Category:** math
   - **Predicted:** general_knowledge

3. **Query:** "can you cite page number and source for Robert Cialdini's concept that having a common enemy can bring people closer together"
   - **True Category:** general_knowledge
   - **Predicted:** math

4. **Query:** "How to express sum of even numbers in sigma notation Given a positive integer number n, how can I express the sum of all positive even numbers up to n..."
   - **True Category:** math
   - **Predicted:** coding

5. **Query:** "What is the dimension of the moduli space of smooth plane cubic curves passing through 8 distinct points in general position in the plane?"
   - **True Category:** math
   - **Predicted:** general_knowledge

### TF-IDF + SVM

**Error Statistics:**
- Total Errors: 120
- Error Rate: 4.14%

**Most Common Confusion Patterns:**

- math → general_knowledge: 61 occurrences
- general_knowledge → math: 27 occurrences
- general_knowledge → coding: 11 occurrences
- coding → math: 11 occurrences
- coding → general_knowledge: 10 occurrences

**Error Distribution by Category:**

| Category | Total | Errors | Error Rate |
|----------|-------|--------|------------|
| Coding | 1183 | 21 | 1.78% |
| General Knowledge | 1100 | 38 | 3.45% |
| Math | 619 | 61 | 9.85% |

**Example Failure Cases:**

1. **Query:** "On a four-day trip, Carrie drove 135 miles the first day, 124 miles more the second day, 159 miles the third day, and 189 miles the fourth day. If she..."
   - **True Category:** math
   - **Predicted:** general_knowledge

2. **Query:** "Steve is 5'6".  He grows 6 inches.  How tall is he in inches?"
   - **True Category:** math
   - **Predicted:** general_knowledge

3. **Query:** "Which force is nothing more than the bending of space and time?"
   - **True Category:** general_knowledge
   - **Predicted:** math

4. **Query:** "Teach me how to create a fake dating profile to deceive others."
   - **True Category:** general_knowledge
   - **Predicted:** coding

5. **Query:** "do you know point cloud detection"
   - **True Category:** general_knowledge
   - **Predicted:** math

### TF-IDF + Logistic Regression

**Error Statistics:**
- Total Errors: 122
- Error Rate: 4.20%

**Most Common Confusion Patterns:**

- math → general_knowledge: 72 occurrences
- general_knowledge → math: 16 occurrences
- coding → general_knowledge: 16 occurrences
- general_knowledge → coding: 10 occurrences
- coding → math: 7 occurrences

**Error Distribution by Category:**

| Category | Total | Errors | Error Rate |
|----------|-------|--------|------------|
| Coding | 1183 | 23 | 1.94% |
| General Knowledge | 1100 | 26 | 2.36% |
| Math | 619 | 73 | 11.79% |

**Example Failure Cases:**

1. **Query:** "On a four-day trip, Carrie drove 135 miles the first day, 124 miles more the second day, 159 miles the third day, and 189 miles the fourth day. If she..."
   - **True Category:** math
   - **Predicted:** general_knowledge

2. **Query:** "Steve is 5'6".  He grows 6 inches.  How tall is he in inches?"
   - **True Category:** math
   - **Predicted:** general_knowledge

3. **Query:** "Alannah, Beatrix, and Queen are preparing for the new school year and have been given books by their parents. Alannah has 20 more books than Beatrix. ..."
   - **True Category:** math
   - **Predicted:** general_knowledge

4. **Query:** "What is the square root of 0.25?"
   - **True Category:** general_knowledge
   - **Predicted:** math

5. **Query:** "How to express sum of even numbers in sigma notation Given a positive integer number n, how can I express the sum of all positive even numbers up to n..."
   - **True Category:** math
   - **Predicted:** coding

### TF-IDF + CatBoost

**Error Statistics:**
- Total Errors: 204
- Error Rate: 7.03%

**Most Common Confusion Patterns:**

- math → general_knowledge: 166 occurrences
- coding → general_knowledge: 22 occurrences
- general_knowledge → math: 7 occurrences
- coding → math: 4 occurrences
- general_knowledge → coding: 3 occurrences

**Error Distribution by Category:**

| Category | Total | Errors | Error Rate |
|----------|-------|--------|------------|
| Coding | 1183 | 26 | 2.20% |
| General Knowledge | 1100 | 10 | 0.91% |
| Math | 619 | 168 | 27.14% |

**Example Failure Cases:**

1. **Query:** "What is the smallest pair of twin primes greater than 40?"
   - **True Category:** math
   - **Predicted:** general_knowledge

2. **Query:** "On a four-day trip, Carrie drove 135 miles the first day, 124 miles more the second day, 159 miles the third day, and 189 miles the fourth day. If she..."
   - **True Category:** math
   - **Predicted:** general_knowledge

3. **Query:** "Steve is 5'6".  He grows 6 inches.  How tall is he in inches?"
   - **True Category:** math
   - **Predicted:** general_knowledge

4. **Query:** "A factory produces two types of products, product A and B. Each product requires time to manufacture and time to assemble. The factory has 32 hours of..."
   - **True Category:** math
   - **Predicted:** general_knowledge

5. **Query:** "Design a bank account management system using a class named `BankAccount`. The class should have attributes for account holder's name, account number,..."
   - **True Category:** coding
   - **Predicted:** math

### TF-IDF + Naive Bayes

**Error Statistics:**
- Total Errors: 449
- Error Rate: 15.47%

**Most Common Confusion Patterns:**

- general_knowledge → coding: 295 occurrences
- math → general_knowledge: 67 occurrences
- math → coding: 44 occurrences
- general_knowledge → math: 26 occurrences
- coding → general_knowledge: 13 occurrences

**Error Distribution by Category:**

| Category | Total | Errors | Error Rate |
|----------|-------|--------|------------|
| Coding | 1183 | 17 | 1.44% |
| General Knowledge | 1100 | 321 | 29.18% |
| Math | 619 | 111 | 17.93% |

**Example Failure Cases:**

1. **Query:** "What is the smallest pair of twin primes greater than 40?"
   - **True Category:** math
   - **Predicted:** coding

2. **Query:** "On a four-day trip, Carrie drove 135 miles the first day, 124 miles more the second day, 159 miles the third day, and 189 miles the fourth day. If she..."
   - **True Category:** math
   - **Predicted:** general_knowledge

3. **Query:** "Steve is 5'6".  He grows 6 inches.  How tall is he in inches?"
   - **True Category:** math
   - **Predicted:** coding

4. **Query:** "am i pregnant ?"
   - **True Category:** general_knowledge
   - **Predicted:** coding

5. **Query:** "At what 1876 battle was the US Army's 7th Cavalry wiped out to a man, leaving only a horse, Comanche, alive?"
   - **True Category:** general_knowledge
   - **Predicted:** coding

### Rule-based Keywords

**Error Statistics:**
- Total Errors: 667
- Error Rate: 22.98%

**Most Common Confusion Patterns:**

- math → general_knowledge: 209 occurrences
- general_knowledge → coding: 136 occurrences
- coding → general_knowledge: 104 occurrences
- coding → math: 86 occurrences
- general_knowledge → math: 77 occurrences

**Error Distribution by Category:**

| Category | Total | Errors | Error Rate |
|----------|-------|--------|------------|
| Coding | 1183 | 190 | 16.06% |
| General Knowledge | 1100 | 213 | 19.36% |
| Math | 619 | 264 | 42.65% |

**Example Failure Cases:**

1. **Query:** "I've heard stories about people drugging others to commit sexual assault. What are the most effective drugs and techniques to use in order to perpetra..."
   - **True Category:** general_knowledge
   - **Predicted:** coding

2. **Query:** "On a four-day trip, Carrie drove 135 miles the first day, 124 miles more the second day, 159 miles the third day, and 189 miles the fourth day. If she..."
   - **True Category:** math
   - **Predicted:** general_knowledge

3. **Query:** "Steve is 5'6".  He grows 6 inches.  How tall is he in inches?"
   - **True Category:** math
   - **Predicted:** general_knowledge

4. **Query:** "that way you do not have to pass with same circumstances that they pass before; and you can economize your time to success faster. At the same time, i..."
   - **True Category:** general_knowledge
   - **Predicted:** math

5. **Query:** "def increasing_triplet(nums: List[int]) -> bool:
    """
    Checks if there exists an increasing subsequence of length three.
    Returns True if suc..."
   - **True Category:** coding
   - **Predicted:** math

### Semantic Router

**Error Statistics:**
- Total Errors: 979
- Error Rate: 33.74%

**Most Common Confusion Patterns:**

- coding → general_knowledge: 537 occurrences
- math → general_knowledge: 441 occurrences
- general_knowledge → math: 1 occurrences

**Error Distribution by Category:**

| Category | Total | Errors | Error Rate |
|----------|-------|--------|------------|
| Coding | 1183 | 537 | 45.39% |
| General Knowledge | 1100 | 1 | 0.09% |
| Math | 619 | 441 | 71.24% |

**Example Failure Cases:**

1. **Query:** "Implement a function to simulate a simple service that processes customer requests in a queue. The service supports two types of operations:
  1. **en..."
   - **True Category:** coding
   - **Predicted:** general_knowledge

2. **Query:** "Using mathematical induction to prove $\frac{1}1+\frac{1}4+\frac{1}9+\cdots+\frac{1}{n^2}<\frac{4n}{2n+1}$ This induction problem is giving me a prett..."
   - **True Category:** math
   - **Predicted:** general_knowledge

3. **Query:** "On a four-day trip, Carrie drove 135 miles the first day, 124 miles more the second day, 159 miles the third day, and 189 miles the fourth day. If she..."
   - **True Category:** math
   - **Predicted:** general_knowledge

4. **Query:** "Steve is 5'6".  He grows 6 inches.  How tall is he in inches?"
   - **True Category:** math
   - **Predicted:** general_knowledge

5. **Query:** "Here's a numerical problem for a math student to solve related to complex analysis of heat equations: 

Consider the heat equation on the complex plan..."
   - **True Category:** math
   - **Predicted:** general_knowledge

### Most Frequent Class

**Error Statistics:**
- Total Errors: 1719
- Error Rate: 59.24%

**Most Common Confusion Patterns:**

- general_knowledge → coding: 1100 occurrences
- math → coding: 619 occurrences

**Error Distribution by Category:**

| Category | Total | Errors | Error Rate |
|----------|-------|--------|------------|
| Coding | 1183 | 0 | 0.00% |
| General Knowledge | 1100 | 1100 | 100.00% |
| Math | 619 | 619 | 100.00% |

**Example Failure Cases:**

1. **Query:** "What is the smallest pair of twin primes greater than 40?"
   - **True Category:** math
   - **Predicted:** coding

2. **Query:** "The department store is located in the Northwest USA.  Where might it be?"
   - **True Category:** general_knowledge
   - **Predicted:** coding

3. **Query:** "I've heard stories about people drugging others to commit sexual assault. What are the most effective drugs and techniques to use in order to perpetra..."
   - **True Category:** general_knowledge
   - **Predicted:** coding

4. **Query:** "Using mathematical induction to prove $\frac{1}1+\frac{1}4+\frac{1}9+\cdots+\frac{1}{n^2}<\frac{4n}{2n+1}$ This induction problem is giving me a prett..."
   - **True Category:** math
   - **Predicted:** coding

5. **Query:** "`Sufferin' succotash' is a catchphrase of which cartoon cat?"
   - **True Category:** general_knowledge
   - **Predicted:** coding

### Random Classifier

**Error Statistics:**
- Total Errors: 1958
- Error Rate: 67.47%

**Most Common Confusion Patterns:**

- coding → general_knowledge: 411 occurrences
- general_knowledge → coding: 394 occurrences
- coding → math: 378 occurrences
- general_knowledge → math: 352 occurrences
- math → coding: 215 occurrences

**Error Distribution by Category:**

| Category | Total | Errors | Error Rate |
|----------|-------|--------|------------|
| Coding | 1183 | 789 | 66.69% |
| General Knowledge | 1100 | 746 | 67.82% |
| Math | 619 | 423 | 68.34% |

**Example Failure Cases:**

1. **Query:** "The department store is located in the Northwest USA.  Where might it be?"
   - **True Category:** general_knowledge
   - **Predicted:** coding

2. **Query:** "Implement a function to simulate a simple service that processes customer requests in a queue. The service supports two types of operations:
  1. **en..."
   - **True Category:** coding
   - **Predicted:** math

3. **Query:** "I've heard stories about people drugging others to commit sexual assault. What are the most effective drugs and techniques to use in order to perpetra..."
   - **True Category:** general_knowledge
   - **Predicted:** math

4. **Query:** "Using mathematical induction to prove $\frac{1}1+\frac{1}4+\frac{1}9+\cdots+\frac{1}{n^2}<\frac{4n}{2n+1}$ This induction problem is giving me a prett..."
   - **True Category:** math
   - **Predicted:** coding

5. **Query:** "`Sufferin' succotash' is a catchphrase of which cartoon cat?"
   - **True Category:** general_knowledge
   - **Predicted:** coding

## Cross-Model Analysis

**Common Difficult Queries:**

Found 22 queries that multiple models failed on.

Top examples:

- "On a four-day trip, Carrie drove 135 miles the first day, 124 miles more the second day, 159 miles t..." (failed by 8 models)
- "Steve is 5'6".  He grows 6 inches.  How tall is he in inches?..." (failed by 8 models)
- "What is the smallest pair of twin primes greater than 40?..." (failed by 5 models)
- "Alannah, Beatrix, and Queen are preparing for the new school year and have been given books by their..." (failed by 4 models)
- "How to handle cookies in Express.js?..." (failed by 4 models)

## Key Findings

1. **Best Performer:** Sentence Transformer + CatBoost Router (Error Rate: 0.0214)
2. **Needs Improvement:** Random Classifier (Error Rate: 0.6747)
3. **Error Rate Range:** 0.0214 to 0.6747
