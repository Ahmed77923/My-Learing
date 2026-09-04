# import pandes as pd
# import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from scipy.stats import uniform

# 1
# Continuous Uniform Distribution    --(وقت انتظار Amir)--
min_time = 0
max_time = 30
# loc = البداية
# scale = النهاية - البداية
wait_dist = uniform(loc=min_time, scale=max_time - min_time)

# أقل من 5 دقائق
prob_less_5 = wait_dist.cdf(5)
print(prob_less_5)

# أكثر من 5 دقائق
prob_more_5 = 1 - wait_dist.cdf(5)
print(prob_more_5)


# بين 10 و 20 دقيقة
prob_between_10_20 = wait_dist.cdf(20) - wait_dist.cdf(10)
print(prob_between_10_20)



# توليد بيانات (Simulation)
wait_times = wait_dist.rvs(size=1000)
print(wait_times[:10])  # أول 10 قيم







# 2 
# Binomial Distribution   --(عدد الصفقات)--
# السيناريو
# كل أسبوع: 3 صفقات   = n
# احتمال النجاح: 30%  = p 
# عدد الأسابيع: 52    =  size


deals = binom.rvs(n=3,p=.3,size=52)
print(deals)



# احتمال إغلاق ≤ صفقة واحدة
prob_le_1 = binom.cdf(1, 3, 0.3)
print(prob_le_1)



# احتمال إغلاق أكثر من صفقة
prob_mo_1 = 1 - binom.cdf(1, 3, 0.3)
print(prob_mo_1)



# Expected Value = Expected = n * p

# if p == 30%
won_30pct = 52 * 0.30
print(won_30pct)


# if p = 35%
won_35pct = 52 * 0.35
print(won_35pct)



# مقارنة سريعة (احفظها)
#  استخدم    الحالة	 
# X = k   → binom.pmf(k, n, p)
# X ≤ k   → binom.cdf(k, n, p)
# X > k   → 1 - binom.cdf(k, n, p)
# simulate → binom.rvs(n, p, size)


c