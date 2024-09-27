# COS method for option pricing
### Description
Implementation of COS method (classic one + model free). \
\
COS.py has implementation of classic COS method. This one was proposed by Fang & Oosterlee (2008). It supports different pricing model: BS, Heston, CGMY (there is no Variance-Gamma because it's basically a CGMY model with $Y = 0$). \
\
iCOS.py has the implementation of option-implied COS method by Vladimirov (2023). It is a model-free variation of classic method.\
### Current goals:
1) Fix interface of iCOS class to make it inline with PricingModel abc.
2) Change characteristic function of Heston model (current one has complex logarithm + square root of complex number $\rightarrow$ numeric instability

