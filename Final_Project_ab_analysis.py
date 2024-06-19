#!/usr/bin/env python
# coding: utf-8

# # Финальный проект
# (Сергеева Юлия)
# 
# ----------

# # Вводные и задача проекта:
# 
# Представьте, что вы работаете в крупном дейтинговом приложении.
# 
# Помимо базовых функций, в приложении также имеется премиум-подписка, которая дает доступ к ряду важных дополнительных возможностей.  
# Был проведен A/B тест, в рамках которого для новых пользователей из нескольких стран была изменена стоимость премиум-подписки* при покупке через две новые платежные системы. При этом стоимость пробного периода оставалась прежней.  
# 
# **Проверьте: был ли эксперимент успешен в целом.**
# 
# *Деньги за подписку списываются ежемесячно до тех пор, пока пользователь её не отменит.
# 
# -------------

# ### Импорт библиотек

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from urllib.parse import urlencode
import requests

from calendar import monthrange
from datetime import timedelta

get_ipython().run_line_magic('matplotlib', 'inline')

from scipy import stats
from scipy.stats import iqr
import pingouin as pg
from scipy.stats import chi2_contingency

from scipy.stats import norm
from tqdm.auto import tqdm


# ### Загрузка файлов

# In[2]:


#внешние ссылки

df_url1 = 'https://disk.yandex.ru/d/4XXIME4osGrMRA'
df_url2 = 'https://disk.yandex.ru/d/yJFydMNNGkEKfg'
df_url3 = 'https://disk.yandex.ru/d/br6KkQupzzTGoQ'
df_url4 = 'https://disk.yandex.ru/d/gvCWpZ55ODzs2g'
df_url5 = 'https://disk.yandex.ru/d/VY5W0keMX5TZBQ'
df_url6 = 'https://disk.yandex.ru/d/th5GL0mGOc-qzg'


# In[3]:


# функция для получения загрузочных ссылок с Яндекс Диска

def get_ds_from_yandex(url):
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    df_final_url = base_url + urlencode(dict(public_key=url))
    response = requests.get(df_final_url)
    download_url = response.json()['href']
    return download_url


# In[4]:


users_test = pd.read_csv(get_ds_from_yandex(df_url1), sep = ';')
users_control_1 = pd.read_csv(get_ds_from_yandex(df_url2), sep = ';')
users_control_2 = pd.read_csv(get_ds_from_yandex(df_url3), sep = ';')
transactions_test = pd.read_csv(get_ds_from_yandex(df_url4), sep = ';')
transactions_control_1 = pd.read_csv(get_ds_from_yandex(df_url5), sep = ';')
transactions_control_2 = pd.read_csv(get_ds_from_yandex(df_url6), sep = ';')


# ### Описание данных:
# 
# Всего есть три группы: 
# * тестовая (test), 
# * контрольная 1 (control_1),
# * контрольная 2 (control_2).  
# 
# Для каждой из них:
# 
# **users_*.csv – информация о пользователях:** 
# 
# uid – идентификатор пользователя  
# age – возраст  
# attraction_coeff – коэффициент привлекательности (от 0 до 1000, лайки / просмотры *1000)  
# 
# coins – число монеток (внутренняя валюта)  
# country – страна    
# visit_days – в какие дни после регистрации пользователь посещал приложение (напр. в 1, затем в 7)  
# gender – пол  
# age_filter_start  – фильтр поиска, мин. значение   
# age_filter_end  – фильтр поиска, макс. значение   
# views_count – число полученных оценок   
# was_premium – был ли когда-либо премиум (либо пробный период премиум-статуса, либо купленный за деньги)  
# is_premium –  является ли премиум  
# total_revenue – нормированная выручка   
# 
# 
# **transactions_*.csv – информация о платежах пользователей:**  
# 
# uid – идентификатор пользователя    
# country – страна  
# joined_at – дата и время регистрации  
# paid_at – дата и время покупки  
# revenue – нормированная выручка  
# payment_id – идентификатор платежа  
# from_page – откуда пользователь перешел на страницу оплаты  
# product_type – тип продукта (trial_premium – пробная премиум-подписка, premium_no_trial – премиум-подписка без пробной, coins – подписка за внутреннюю валюту, other_type – другое)   
# 
# #### Файлы:
# 
# users_test – информация о пользователях в тестовой группе;   
# users_control_1 – информация о пользователях в первой контрольной группе;   
# users_control_2 – информация о пользователях во второй контрольной группе;     
# transactions_test – информация о платежах пользователей в тестовой группе;   
# transactions_control_1 – информация о платежах пользователей в первой контрольной группе;   
# transactions_control_2 – информация о платежах пользователей во второй контрольной группе.   

# ----------
# 
# # Этапы работы:
# - Разведочный анализ исходных данных (EDA) (строки 5 - 107);
# - Проверка групп эксперимента на репрезентативность (строки 108 - 110);
# - Выбор и обоснование метрик;
# - Сравнение метрик в рамках A/A и A/B тестирования и оценка статистической значимости полученных результатов (строки 116 - 155);
# - Выводы по результатам анализа.

# -----------------------
# # Разведочный анализ данных (EDA)

# Перед обработкой данных и непосредственно анализом этих данных объединим их, но не будем спешить объединять все таблицы в одну (данные по отдельности могут пригодиться в дальнейшем), а пока объединим таблицы по группам (инфо о пользователях + инфо об их платежах).

# In[5]:


# users_test – информация о пользователях в тестовой группе:

users_test.head()


# In[6]:


# Размер таблицы:

users_test.shape


# In[7]:


# Количество уникальных id в ней:

users_test.uid.nunique()


# In[8]:


# transactions_test – информация о платежах пользователей в тестовой группе:

transactions_test.head()


# In[9]:


# Размер таблицы:

transactions_test.shape


# In[10]:


# Количество уникальных id в ней:

transactions_test.uid.nunique()


# In[11]:


# Список уникальных id пользователей из таблицы users_test:
t = users_test.uid.unique().tolist()


# Проверяем, есть ли в таблице transactions_test id пользователей, которых нет в users_test:
transactions_test .query('uid not in @t')


# In[12]:


# Таковых нет, поэтому просто применим ниже left join:


# Объединенные данные по тестовой группе:
# 

# In[13]:


test_group = pd.merge(users_test, transactions_test, on = 'uid', how='left')

test_group


# -----------------------------

# In[14]:


# Список уникальных id пользователей из таблицы users_control_1:
t2 = users_control_1.uid.unique().tolist()


# Проверяем, есть ли в таблице transactions_control_1 id пользователей, которых нет в users_control_1:
transactions_control_1 .query('uid not in @t2')


# In[15]:


transactions_control_1 .query('uid not in @t2') .info()


# In[16]:


transactions_control_1['joined_at'] = pd.to_datetime(transactions_control_1['joined_at'], format='%Y-%m-%d %H:%M:%S')
transactions_control_1['paid_at'] = pd.to_datetime(transactions_control_1['paid_at'], format='%Y-%m-%d %H:%M:%S')

transactions_control_1 .query('uid not in @t2') .info()


# In[17]:


# Такие есть, но все нулевые/пустые, смело можем от них отказаться и тоже применить left join:


# Объединенные данные по контрольной группе 1:
# 

# In[18]:


control_group_1 = pd.merge(users_control_1, transactions_control_1, on = 'uid', how='left')

control_group_1


# ---------------------------

# In[19]:


# Список уникальных id пользователей из таблицы users_control_2:
t3 = users_control_2.uid.unique().tolist()


# Проверяем, есть ли в таблице transactions_control_1 id пользователей, которых нет в users_control_2:
transactions_control_2 .query('uid not in @t3')


# In[20]:


# Таковых нет, поэтому также применим ниже left join:


# Объединенные данные по контрольной группе 2:
# 

# In[21]:


control_group_2 = pd.merge(users_control_2, transactions_control_2, on = 'uid', how='left')

control_group_2


# ----------------------------

# **Изучим и обработаем полные данные тестовой группы, преобразуем их типы при необходимости:**
# 

# In[22]:


# размер таблицы:

test_group.shape


# In[23]:


# информацию о типах данных и их ненулевых значениях в таблице:

test_group.info()


# In[24]:


# количество пустых значений в таблице: 

test_group.isna().sum()


# **В тестовой группе есть множество пользователей с пустым значением по колонке visit_days (в какие дни после регистрации пользователь посещал приложение), посмотрим на этих пользователей подробнее, возможно, это те, кто зарегистрировался и больше не заходил в приложение?**

# In[25]:


test_group[test_group.visit_days.isna()].query('payment_id != "nan"').head()


# **Видим, что среди них есть те, кто платил за подписку (paid_at) задолго после регистрации (joined_at), то есть были на платформе, как минимум, 2 дня, что значит, что колонка visit_days немного "бракованная" :(** 
# 
# **Будем иметь это в виду при дальнейшем анализе.**

# Также почему-то payment_id не уникальные ни по датам покупок, ни по id пользователя.
# И никакой закономерности в присвоении payment_id не обнаружено. И судя по описанию данных, payment_id != платежная система.
# 
# Также будем иметь это в виду, не опираясь на эту колонку при расчетах.
# 
# -----------------

# In[26]:


# на всякий случай приводим часть колонок типа object к str во избежание дальнейших ошибок в расчетах: 

test_group.uid = test_group.uid.apply(str)
test_group.gender= test_group.gender.apply(str)
test_group.payment_id = test_group.payment_id.apply(str)
test_group.product_type = test_group.product_type.apply(str)


# и даты к датам: 

test_group['joined_at'] = pd.to_datetime(test_group['joined_at'], format='%Y-%m-%d %H:%M:%S')
test_group['paid_at'] = pd.to_datetime(test_group['paid_at'], format='%Y-%m-%d %H:%M:%S')


# In[27]:


# Найдем первые и последние даты в таблице. 
# по дате регистрации:
test_group.joined_at.describe(datetime_is_numeric = True) 


# In[28]:


# по дате покупки:
test_group.paid_at.describe(datetime_is_numeric = True) 


# In[29]:


# Во тестовой группе все данные за 2017 год.


# In[30]:


# Проверим, нет ли ошибок в данных, например, случаев, где дата покупки идет раньше, чем дата регистрации: 

test_group[test_group.paid_at < test_group.joined_at]


# In[31]:


# Такие кейсы нашлись, посчитаем их кол-во по уникальным id:

test_group[test_group.paid_at < test_group.joined_at].uid.nunique()


# In[32]:


# нашлось 35 uid (только в тестовой группе). 
# В этих кейсах подозрительно то, что дата покупки во всех случаях - 11е число какого-либо месяца.
# Возможно, тут перепутаны местами дата и месяц?
# Посмотрим на список дат оплаты, где число = 11 : 

test_group[test_group.paid_at.dt.day == 11].paid_at.unique()


# В записях вида YYYY-MM-DD  MM не превышает 12, что снимает подозрение о том, что на место MM у нас ошибочно попали дни (числа месяца).  
# Вероятно, имела место ошибка в записи данных о дате регистрации / дате оплаты.  
# 
# По-хорошему, даже если мы исключим записи о таких юзерах (пытаясь получить корректные данные только о новых пользователях
# в анализе), это не гарантирует нам отсутствие ошибок в оставшихся данных и наличие только-новых пользователей
# (с учетом того, что дата решистрации известна только по пользователям с покупками).  
# 
# Хорошо бы найти и поправить этот баг, но не имея такой возможности, будем считать, что баг только в конкретно этих записях.  
# 
# Ниже исключим эти id из анализа.

# In[33]:


test_date_errors_id_list = test_group[test_group.paid_at < test_group.joined_at].uid.unique()


# In[34]:


# Перезаписываем таблицу тестовой группы, исключая юзеров, по которым нашлись багованные данные по датам:

test_group = test_group.query('uid not in @test_date_errors_id_list')


# ---------------------
# **Таким же образом изучим и обработаем полные данные первой контрольной группы, преобразуем их типы при необходимости:**
# 

# In[35]:


# размер таблицы:

control_group_1.shape


# In[36]:


# информацию о типах данных и их ненулевых значениях в таблице:

control_group_1.info()


# In[37]:


# количество пустых значений в таблице: 

control_group_1.isna().sum()


# **В контрольной группе № 1 ситуация с описанием данных аналогичная.**

# In[38]:


# Также приводим часть колонок типа object к str во избежание дальнейших ошибок в расчетах: 

control_group_1.uid = control_group_1.uid.apply(str)
control_group_1.gender= control_group_1.gender.apply(str)
control_group_1.payment_id = control_group_1.payment_id.apply(str)
control_group_1.product_type = control_group_1.product_type.apply(str)


# и даты к датам: 

control_group_1['joined_at'] = pd.to_datetime(control_group_1['joined_at'], format = '%Y-%m-%d %H:%M:%S')
                                              
control_group_1['paid_at'] = pd.to_datetime(control_group_1['paid_at'], format = '%Y-%m-%d %H:%M:%S')


# In[39]:


# Также найдем первые и последние даты в таблице. 
# по дате регистрации:
control_group_1.joined_at.describe(datetime_is_numeric = True)           


# In[40]:


# по дате покупки:
control_group_1.paid_at.describe(datetime_is_numeric = True)           


# In[41]:


# данные контрольной группы №1 без учета пустых значений по joined_at:

control_group_1.query('joined_at != "NaT"').sort_values('joined_at').head()


# In[42]:


# в первой контрольной группе есть 1 юзер из России, зарегистрировавшийся в 2015 году с покупками в 2016 году, 
# все остальные данные - за 2017 год.
# При этом у этого юзера относительно низкие чеки по этим покупка, и судя по from_page = refund_VP, это какие-то возраты.
# К тому же, по нему неверно посчитан total revenue (c другими пользователями такой проблемы не наблюдается).

# На всякий случай исключим этого юзера из таблицы:


# In[43]:


control_group_1 = control_group_1.query('uid != "960936960"')


# In[44]:


# описательная статистика по датам в столбце joined_at:

control_group_1.joined_at.describe(datetime_is_numeric = True)           


# In[45]:


# описательная статистика по датам в столбце paid_at:

control_group_1.paid_at.describe(datetime_is_numeric = True)   


# In[46]:


control_1_date_errors_id_list = control_group_1[control_group_1.paid_at < control_group_1.joined_at].uid.unique()

# Перезаписываем таблицу контрольной группы №1, исключая юзеров, по которым нашлись багованные данные по датам:

control_group_1 = control_group_1.query('uid not in @control_1_date_errors_id_list')


# ---------------------
# **Таким же образом изучим и обработаем полные данные второй контрольной группы, преобразуем их типы при необходимости:**
# 

# In[47]:


# размер таблицы:

control_group_2.shape


# In[48]:


# информацию о типах данных и их ненулевых значениях в таблице:

control_group_2.info()


# In[49]:


# количество пустых значений в таблице: 

control_group_2.isna().sum()


# **И в контрольной группе № 2 ситуация с описанием данных аналогичная.**

# In[50]:


# Также приводим часть колонок типа object к str во избежание дальнейших ошибок в расчетах: 

control_group_2.uid = control_group_2.uid.apply(str)
control_group_2.gender= control_group_2.gender.apply(str)
control_group_2.payment_id = control_group_2.payment_id.apply(str)
control_group_2.product_type = control_group_2.product_type.apply(str)


# и даты к датам: 

control_group_2['joined_at'] = pd.to_datetime(control_group_2['joined_at'], format='%Y-%m-%d %H:%M:%S')
control_group_2['paid_at'] = pd.to_datetime(control_group_2['paid_at'], format='%Y-%m-%d %H:%M:%S')


# In[51]:


# Также найдем первые и последние даты в таблице. 
# по дате регистрации:
control_group_2.joined_at.describe(datetime_is_numeric = True) 


# In[52]:


# по дате покупки:
control_group_2.paid_at.describe(datetime_is_numeric = True)      


# In[53]:


# Во второй контрольной группе все данные за 2017 год.


# Итого, во всех таблицах   
# первая дата регистрации - 2017-01-11, последняя - 2017-10-31,
# первая дата покупки - 2017-01-11, последняя - 2017-12-11.
# 
# То есть эксперимент проходил в течение года, что дает нам дополнительную уверенность в том, что сезонный фактор не подпортил нам статистику, а также косвенно указывает на то, что была набрана стат. мощность (долго копили нужный объем данных).
# 
# 
# НО важно учесть и помнить, что значения по датам регистрации и покупки не-пустые только у пользователей, которые совершали покупки! По остальным юзерам нет даты регистрации.

# In[54]:


control_2_date_errors_id_list = control_group_2[control_group_2.paid_at < control_group_2.joined_at].uid.unique()

# Перезаписываем таблицу контрольной группы №2, исключая юзеров, по которым нашлись багованные данные по датам:

control_group_2 = control_group_2.query('uid not in @control_2_date_errors_id_list')


# ---------------------------

# In[55]:


# Проверим значимость колонки was_premium для нашего анализа 
# (можно ли по ней ориентироваться на пользователей как на тех, кто купил премиум впервые)

test_uids_with_prem_were_prem = test_group .query('was_premium == "1" and product_type == "trial_premium" or product_type == "premium_no_trial"') .uid .to_list()


# список пользователей из тестовой группы, у которых была премиальная покупка и есть статус "был когда-либо премиум"
test_group.query('uid in @test_uids_with_prem_were_prem') .groupby('uid') .agg({'product_type': 'count'}) .sort_values('product_type', ascending = False)


# In[56]:


# посмотрим на список покупок одного из таких пользователей в хронологическом порядке:

test_group.query('uid == "892236423"').sort_values('paid_at')


# **Видим, что даже в день покупки премиума (в день регистрации), у него уже есть статус "был когда-либо премиум".**
# 
# 
# (Вообще в целом лишний шаг, т.к. это логично ввиду того, что was_premium - данные из таблицы с уникальными id, 
# другого исхода и не стоило ждать, но лучше перебдеть).
# 
# **Итого, колонка was_premium нам в анализе никак не поможет.**
# 

# In[57]:


# Также обратим внимание на статус is_premium –  является премиум.
# Для проверки данных по этой колонке объединим все данные в одну таблицу.


# Объединим данных трех групп, участвующих в тесте, в одну таблицу, предварительно добавив в каждую из них столбец, обозначающий группу, присвоенную каждому юзеру:

# In[58]:


test_group["group"] = 'test'
control_group_1["group"] = 'control_1'
control_group_2["group"] = 'control_2'


# In[59]:


df_merged = test_group.append(control_group_1, sort = False).append(control_group_2, sort = False)


# In[60]:


# список пользователей, которые покупали премиум или пробный премиум:
bought_premium_uids = df_merged.query('product_type == "premium_no_trial" or product_type == "trial_premium"').uid.unique().tolist()


# уберем этих пользователей из тотал-данных и оставим только тех, у кого статус "is_premium":
df_merged.query('is_premium == 1 and uid not in @bought_premium_uids')


# **Видим, что в данных есть 132 записи о пользователях, у которых статус говорит о их премиальности, и при этом в рамках нашего эксперимента они не покупали премиум.**
# 
# То есть на самом деле в эксперименте участвовали не только новые пользователи?
# Либо есть баг в присвоении значения is_premium?
# 
# Узнать наверняка, какое из предположений верное, мы сейчас не можем. 
# 

# In[61]:


# скольких уникальных пользователей это зааффектило?
df_merged.query('is_premium == 1 and uid not in @bought_premium_uids').uid.nunique()


# In[62]:


df_merged.query('is_premium == 1 and uid not in @bought_premium_uids').query('paid_at != "NaT"').uid.nunique()


# In[63]:


# 120 пользователей, 9 из которых совершали покупки в рамках эксперимента.


# **Но предположим, что верно второе предположение (баг в присвоении значения is_premium). В таком случае мы просто не будем опираться на столбец is_premium при дальнейшем анализе.**
# 
# **В случае же, если верно утверждение, что в эксперименте участвовали не только новые пользователи, исключив этих 120 пользователей мы не гарантируем избавление от всех "старых" пользователей.**
# 
# 

# ----------------------
# **Посчитаем среднюю стоимость подписки по новым пользователям по каждой группе, тем самым убедимся, что стоимость премиума в тестовой группе действительно отличается.**   
# 
# По идеи в двух контрольных группах цена должна быть одинаковой, а в тестовой - должна отличаться. Проверим.

# In[64]:


# во всех трех таблицах ниже удаляем дубликаты по столбцу uid после сортировки по дате оплаты, 
# т.е. оставляем только первые оплаты пользователя + всех тех, кто ничего не покупал (они должны быть и так уникальные), 
# и фильтруем по продукту, оставляем только премиум без пробного периода:


# In[65]:


test_group_premium_newusers = test_group.sort_values('paid_at').drop_duplicates(subset=['uid']).query('product_type == "premium_no_trial"')

# Цена за премиум-подписку для новых пользователей в тестовой группе:
round(test_group_premium_newusers.revenue.sum() / test_group_premium_newusers.product_type.count(), 2)


# In[66]:


control_group_1_premium_newusers = control_group_1.sort_values('paid_at').drop_duplicates(subset=['uid']).query('product_type == "premium_no_trial"')


# Цена за премиум-подписку для новых пользователей в контрольной группе №1:
round(control_group_1_premium_newusers.revenue.sum() / control_group_1_premium_newusers.product_type.count(), 2)


# In[67]:


control_group_2_premium_newusers = control_group_2.sort_values('paid_at').drop_duplicates(subset=['uid']).query('product_type == "premium_no_trial"')


# Цена за премиум-подписку для новых пользователей в контрольной группе №2:
round(control_group_2_premium_newusers.revenue.sum() / control_group_2_premium_newusers.product_type.count(), 2)


# Видим, что цена премиума в двух контрольных группах различается (хотя по логике в контрольных группах она должна была бы остаться одинаковой).

# **Возможно, цена премиума в контрольных группах не одинаковая из-за применения промо? ( promo_09 в from_page)**

# **Исключим promo_09 и посмотрим на среднюю стоимость подписки без учета промо:**
# 

# In[68]:


test_group_premium_newusers_2 = test_group.sort_values('paid_at').drop_duplicates(subset=['uid']).query('product_type == "premium_no_trial" and from_page != "promo_09"')

# Цена за премиум-подписку для новых пользователей в тестовой группе:
test_group_premium_newusers_2.revenue.sum() / test_group_premium_newusers_2.product_type.count()


# In[69]:


control_group_1_premium_newusers_2 = control_group_1.sort_values('paid_at').drop_duplicates(subset=['uid']).query('product_type == "premium_no_trial" and from_page != "promo_09"')


# Цена за премиум-подписку для новых пользователей в контрольной группе №1:
control_group_1_premium_newusers_2.revenue.sum() / control_group_1_premium_newusers_2.product_type.count()


# In[70]:


control_group_2_premium_newusers_2 = control_group_2.sort_values('paid_at').drop_duplicates(subset=['uid']).query('product_type == "premium_no_trial" and from_page != "promo_09"')


# Цена за премиум-подписку для новых пользователей в контрольной группе №2:
control_group_2_premium_newusers_2.revenue.sum() / control_group_2_premium_newusers_2.product_type.count()


# **Разница в цене, как и сами средние цены, в контрольных группах осталась +- прежней,  то есть взаимодействие со страницей promo_09 не предполагало снижение стоимость подписки.**
# 
# **Посмотрим, что еще могло влиять на прайс в контрольных группах:**

# ---------------------
# **Посчитаем среднюю цену премиума в контрольных группах в разрезе по странам и по страницам.**
# 

# In[71]:


# Контрольная группа №1:
    
f = control_group_1_premium_newusers.groupby(['country_y', 'from_page', 'revenue'], as_index = False).agg({'product_type': 'count'})
f['price'] = round(f.revenue / f.product_type, 2)

f


# In[72]:


# Контрольная группа №2:

f2 = control_group_2_premium_newusers.groupby(['country_y', 'from_page', 'revenue'], as_index = False).agg({'product_type': 'count'})
f2['price'] = round(f.revenue / f.product_type, 2)

f2


# **Внутри одной группы, в рамках одной страны, цена была разной, даже в разрезе одной посещенной страницы.**  
# **Но на это мог повлиять курс валют, например.**

# **Возможно, прайс менялась от месяца к месяцу?**
# 

# In[73]:


# посмотрим на прайсы по месяцам в первой контрольной группе:

g1 = control_group_1_premium_newusers
g1["payment_month"] = g1['paid_at'].dt.to_period("M")


# In[74]:


gg1 = g1.groupby(['payment_month', 'revenue'], as_index = False).agg({'product_type': 'count'})

gg1['price'] = round(gg1.revenue / gg1.product_type, 2)

gg1.sort_values('revenue')


# In[75]:


# видим по первой контрольной группе значительные выбросы в апреле и октябре, посмотрим на эти месяцы детальнее:


# In[76]:


g1.query('paid_at > "2017-09-30" and paid_at < "2017-11-01" or paid_at > "2017-03-31" and paid_at < "2017-05-01"').sort_values('revenue', ascending = False)


# In[77]:


# нашли 2 uid с аномальной суммой оплаты (63037.0): 


# In[78]:


control_group_1.query('uid == "892216461" or uid == "891383310"')


# In[79]:


# один из них из Турции совершил единственную оплату в день регистрации,
# а второй - из США, после оплаты премиума еще внес доп. оплату коинами в тот же день позже.


# In[80]:


# Чуть ниже исключим эти id из анализа и еще раз посмотрим на стоимость подписки.


# In[81]:


# А пока проделаем то же самое упражнение по второй контрольной группе:


# In[82]:


g2 = control_group_2_premium_newusers
g2["payment_month"] = g2['paid_at'].dt.to_period("M")


gg2 = g2.groupby(['payment_month', 'revenue'], as_index = False).agg({'product_type': 'count'})

gg2['price'] = round(gg2.revenue / gg2.product_type, 2)

gg2.sort_values('revenue')


# In[83]:


# В контрольной группе №2 также есть выбросы в октябре, найдем id:


# In[84]:


g2.query('paid_at > "2017-09-30" and paid_at < "2017-11-01"').sort_values('revenue', ascending = False)


# In[85]:


# тут тоже нашли 2 uid с анамальной суммой оплаты. 


# In[86]:


control_group_2.query('uid == "891778551" or uid == "892307238"')


# In[87]:


# один из них из UAE совершил единственную оплату в день регистрации,
# а второй - из Испании, после оплаты премиума еще внес доп. оплату коинами в тот же день минутой пойзже.


# In[88]:


# Ну и на всякий случай таким же образом проверим и тестовую группу:

g3 = test_group_premium_newusers
g3["payment_month"] = g3['paid_at'].dt.to_period("M")


gg3 = g3.groupby(['payment_month', 'revenue'], as_index = False).agg({'product_type': 'count'})

gg3['price'] = round(gg3.revenue / gg3.product_type, 2)

gg3.sort_values('revenue')


# In[89]:


# В тестовой группе также найден выброс по revenue - в сентябре (113477.0), найдем id:

g3.query('paid_at > "2017-08-31" and paid_at < "2017-10-01"').sort_values('revenue', ascending = False)


# In[90]:


test_group.query('uid == "891178380"')


# In[91]:


# Юзер из Испании, совершил оплату премиума 11го сентября и через 2 месяца совершил еще 2 покупки в одно и то же время.


# In[92]:


# Исключим из анализа подозрительные id из всех трех групп и еще раз посмотрим на стоимость подписки:


# In[93]:


suspicious_uids = ['892216461', '891383310', '891778551', '892307238', '891178380'] 


# In[94]:


control_group_1 = control_group_1.query('uid not in @suspicious_uids') 

control_group_1_premium_newusers_3 = control_group_1.sort_values('paid_at').drop_duplicates(subset=['uid']).query('product_type == "premium_no_trial"')


# Цена за премиум-подписку для новых пользователей в контрольной группе №1:
control_group_1_premium_newusers_3.revenue.sum() / control_group_1_premium_newusers_3.product_type.count()


# In[95]:


control_group_2 = control_group_2.query('uid not in @suspicious_uids') 


control_group_2_premium_newusers_3 = control_group_2.sort_values('paid_at').drop_duplicates(subset=['uid']).query('product_type == "premium_no_trial"')


# Цена за премиум-подписку для новых пользователей в контрольной группе №2:
control_group_2_premium_newusers_3.revenue.sum() / control_group_2_premium_newusers_3.product_type.count()


# In[96]:


test_group = test_group.query('uid not in @suspicious_uids') 


test_group_premium_newusers_3 = test_group.sort_values('paid_at').drop_duplicates(subset=['uid']).query('product_type == "premium_no_trial"')

# Цена за премиум-подписку для новых пользователей в тестовой группе:
test_group_premium_newusers_3.revenue.sum() / test_group_premium_newusers_3.product_type.count()


# Итоговая разница в средней стоимости премиум-подписки между контрольными группами - 540 руб (~10%) и может быть объяснима разницей в курсах валют в момент оплаты.
# 
# Средняя стоимость премиум-подписки в тестовой группе на 55% выше, чем в контрольной группе №1 и на 40% выше, чем в контрольной группе №2.
# 
# Таким образом, мы убедились, что есть значительное изменение стоимости премиум-подписки в тестовой группе относительно контрольных.
# 

# -------------------------
# **Проанализируем картину по странам немного в другом разрезе.**  
# 
# Стоимость подписки была изменена для новых пользователей **из нескольких стран**,  
# проверим, совпадают ли страны по группам и какой порядок прайсов для новых пользователей по странам.

# In[97]:


test_group_countries = test_group_premium_newusers_3.groupby('country_x', as_index = False).agg({'revenue': 'sum', 'product_type': 'count'})
test_group_countries['price'] = round(test_group_countries.revenue / test_group_countries.product_type, 2)
test_group_countries = test_group_countries.rename(columns = {'price': 'price_test'})

test_group_countries


# In[98]:


control_group_1_countries = control_group_1_premium_newusers_3.groupby('country_x', as_index = False).agg({'revenue': 'sum', 'product_type': 'count'})
control_group_1_countries['price'] = round(control_group_1_countries.revenue / control_group_1_countries.product_type, 2)
control_group_1_countries = control_group_1_countries.rename(columns = {'price': 'price_control_1'})

control_group_1_countries


# In[99]:


control_group_2_countries = control_group_2_premium_newusers_3.groupby('country_x', as_index = False).agg({'revenue': 'sum', 'product_type': 'count'})
control_group_2_countries['price'] = round(control_group_2_countries.revenue / control_group_2_countries.product_type, 2)
control_group_2_countries = control_group_2_countries.rename(columns = {'price': 'price_control_2'})

control_group_2_countries


# In[100]:


# Списки стран в группах эксперимента немного отличаются.

# Объединим данные по странам в одну таблицу:


# In[101]:


countries_merged = pd.merge(control_group_1_countries, control_group_2_countries, on = 'country_x', how='outer')
countries_merged = pd.merge(countries_merged, test_group_countries, on = 'country_x', how='outer')

countries_merged = countries_merged[["country_x", "price_control_1", "price_control_2", "price_test"]]

countries_merged


# In[102]:


# График распределения цен по странам в разрезе по группам эксперимента:

sns.set(
         rc={'figure.figsize':(24,12)}
        )


countries_merged[['country_x', 'price_control_1', 'price_control_2', 'price_test']] .plot(x='country_x', kind='bar', rot=75, fontsize=20) 


# In[103]:


# есть странные изменения цен между группами по некоторым странам, посчитаем изменения в процентах:


# In[104]:


countries_merged['change_rate_c2_to_c1'] = round((countries_merged.price_control_2 / countries_merged.price_control_1 - 1) * 100, 2)
countries_merged['change_rate_test_to_c1'] = round((countries_merged.price_test / countries_merged.price_control_1 - 1) * 100, 2)
countries_merged['change_rate_test_to_c2'] = round((countries_merged.price_test / countries_merged.price_control_2 - 1) * 100, 2)

countries_merged


# ----------------
# **Будем считать, что изменение цены до ~20% является погрешностью курса валют.**  
# 
# **Что имеем с учетом этого:**  
# 
# **Значительная разница в цене между контрольными группами (нечистый эксперимент):**   
# Australia, Germany, Italy, United Kingdom (Great Britain).
# 
# **Нет или почти нет изменений в тестовой группе относительно хотя бы одной из контрольных:**  
# Turkey, Germany, United Kingdom (Great Britain).
# 
# **снижение цены в тесте относительно контрольных групп:**  
# United Kingdom (Great Britain)  
# (в целом снижение цены удовлетворяет условиям теста, т.к. по вводным цена "изменилась", не обязательно увеличилась, но 1) изменения близко к 20% в тесте относительно контрольной группы 1 и близко к нулю относительно контрольной группы 2. 2) есть значительная разница в стоимости между контрольными группами. Не будем брать эту страну в расчет).    
# 
# **Тест не задался (не было новых пользователей с покупкой премиума в тестовой группе)**:   
# Australia, Belgium, India, Portugal, Switzerland, Ukraine.  
#     
# **Не было новых пользователей с покупкой премиума в обеих контрольных группах:**   
# Mexico.
# 
# 
# **Итого, далее будем исключать из анализа следующие страны:**  
# 
# Australia, Belgium, Germany, Italy, Turkey, United Kingdom (Great Britain), India, Portugal, Switzerland, Ukraine, Mexico (11 стран из 19 стран, в которых были покупки премиум-подписки без пробника новыми пользователями).
# 
# -----------------
# 
# 
# 

# In[105]:


country_list = ['Australia', 'Belgium', 'Germany', 'Italy', 'Turkey', 'United Kingdom (Great Britain)', 'India', 'Portugal', 'Switzerland', 'Ukraine', 'Mexico']


# In[106]:


df_merged = df_merged.query('country_x not in @country_list and uid not in @suspicious_uids')


# -----------------------
# ### Резюме по разведочному анализу:
# 
# - Конкретная дата регистрации в данных доступна только по тем пользователям, которые совершили покупку;  
# - В вводных имеется инфо, что измененная цена премиум-подписки предлагалась именно новым пользователям. Т.к. нет очевидного указателя на "новизну" пользователя, будем считать, что оставшиеся после обработки пользователи, попавших в группы эксперимента, для нашего сервиса являются новыми, попавшими в базу впервые;
# 
# - Также у нас нет данных о типах платежных систем, про которые упоминается в задаче, этой переменной мы также не сможем оперировать и будем считать, что в тесте только наши две платежные системы и участвовали;  
# - Столбцы was_premium, is_premium и payment_id, visit_days - не используем в анализе (но хотелось бы перепроверить их на баги в рамках отдельной задачи);
# 
# - Дополнительно будем учитывать, что экспреримент проводился в течение года (2017-01-11 - 2017-12-11), а значения по датам регистрации и покупки не-пустые только у пользователей, которые совершали покупки (по остальным юзерам нет даты регистрации).
# 
# - В ходе анализа были выявлены несколько записей о пользователях с аномальными суммами транзакций на исключение из дальнейшего анализа, а также данные по нескольким странам на исключение из дальнейшего анализа (по странам - основываясь на предположении о курсе валют и на данных с аномальной разницей в стоимости премиума в контрольных группах).  
# 
# - **В результате предобработки данных было подтверждено наличие изменений в стоимости премиум-подписки в тестовой группе. Ниже приведена итоговая таблица со средними значениями этой метрики по группам эксперимента.**

# In[107]:


pd.DataFrame({'Группа': 
              ['Контрольная 1', 'Контрольная 2', 'Тестовая'],
              'Цена премиум-подписки': 
              [round(control_group_1_premium_newusers_3.query('country_x not in @country_list').revenue.sum() / control_group_1_premium_newusers_3.query('country_x not in @country_list').product_type.count(), 2), 
               round(control_group_2_premium_newusers_3.query('country_x not in @country_list').revenue.sum() / control_group_2_premium_newusers_3.query('country_x not in @country_list').product_type.count(), 2), 
               round(test_group_premium_newusers_3.query('country_x not in @country_list').revenue.sum() / test_group_premium_newusers_3.query('country_x not in @country_list').product_type.count(), 2)]})


# ---------------------
# 
# # Проверка групп эксперимента на репрезентативность

# Проверим уже очищенные ранее данные.  
# Задачей проверки групп на репрезентативность является выявление возможных отклонений (выбросов) в количестве юзеров, разбитых по различным подгруппам. Отсутствие значительных девиаций по таким подгруппам (возраст, страна, пол), будет свидетельствовать о релевантности состава групп для проведения тестирования.

# In[108]:


# Распределение юзеров по группам по странам: 

country_test = test_group.groupby('country_x', as_index=False).agg({'uid' : 'count'})
country_cntr_1 = control_group_1.groupby('country_x', as_index=False).agg({'uid' : 'count'})
country_cntr_2 = control_group_2.groupby('country_x', as_index=False).agg({'uid' : 'count'})
country = country_test.merge(country_cntr_1, on = 'country_x').merge(country_cntr_2, on = 'country_x')
country = country.rename(columns={'uid_x' : 'test', 'uid_y' : 'control_1', 'uid' : 'control_2'})

country.sort_values('test', ascending=False)


# In[109]:


# между группами аномалий в количестве юзеров по странам нет, переходим к 

# оценке распределения юзеров по группам по полу:

gender_test = test_group.groupby('gender', as_index=False).agg({'uid' : 'count'})
gender_control_1 = control_group_1.groupby('gender', as_index=False).agg({'uid' : 'count'})
gender_control_2 = control_group_2.groupby('gender', as_index=False).agg({'uid' : 'count'})
gender = gender_test.merge(gender_control_1, on = 'gender').merge(gender_control_2, on = 'gender')         .rename(columns={'uid_x' : 'test', 'uid_y' : 'control_1', 'uid' : 'control_2'})

gender


# In[110]:


# между группами аномалий в количестве юзеров по полу нет, переходим к 

# оценке распределения юзеров по группам по возрасту:

age_test = test_group.groupby('age', as_index=False).agg({'uid' : 'count'})
age_cntr_1 = control_group_1.groupby('age', as_index=False).agg({'uid' : 'count'})
age_cntr_2 = control_group_2.groupby('age', as_index=False).agg({'uid' : 'count'})
age = age_test.merge(age_cntr_1, on = 'age').merge(age_cntr_2, on = 'age')       .rename(columns={'uid_x' : 'test', 'uid_y' : 'control_1', 'uid' : 'control_2'})

plt.figure(figsize=(12, 7))
plt.plot(age.age, age.test, label = 'test')
plt.plot(age.age, age.control_1, label = 'control_1')
plt.plot(age.age, age.control_2, label = 'control_2')
plt.title('Распределение количества пользователей по возрастам')
plt.xlabel('Возраст')
plt.ylabel('Количество пользователей')
plt.legend()
plt.grid(True)


# **С составом групп по странам, полу и возрасту все ОК, группы репрезентативны, можем приступать к анализу метрик.**

# ----------------------
# ### Вернемся к задаче проекта:  
# 
# -----------------
# 
# **Какова цель премиум-подписки для бизнеса?**  
# Основная цель - повышение выручки  
# Дополнительная - удержание пользователей (как следствие - лояльность)
# 
# 
# **Какое преимущество дает премиум-подписка пользователю?**  
# В задании нет прямого ответа на этот вопрос, есть лишь формулировка, что подписка "дает доступ к ряду важных дополнительных возможностей". Можно предположить, что это бОльшее число показов профиля пользователя (другим пользователям), выведение в топ выдачи (как конечная цель - больше мэтчей/взаимодействий).
# 
# ### Какова цель A/B теста?  
# Проверить, сможем ли мы привлекать больше выручки, повысив стоимость подписки  
# aka  
# не уроним ли мы конверсию в покупку подписки, предлагая новым клиентам более высокую стоимость подписки  
# 
# 
# ### **Задачи анализа:**  
# **Проверить, был ли эксперимент успешен в целом**, то есть:
# - корректно ли он проведен
# - можно ли назвать результат эксперимента удовлетворительным  
# (если конверсия и/или Retention значимо снизились, то на долгосрочной перспективе увеличенная стоимость подписки (средний чек) может отразиться негативно на показателях бизнеса и не привести к увеличению выручки, даже если в моменте (в рамках эксперимента) сама по себе выручка увеличилась.
# 
# 
# # Какие метрики будем проверять?
# * Конверсию в покупку премиума
# * Выручку на пользователя (ARPU тотал по всем продуктам и отдельно по премиуму)
# * Повторные покупки премиума  
# 
# ---------------
# 

# Перед проведением A/A- и A/B-тестов рассмотрим значения этих метрик по группам эксперимента:

# **Конверсия в первую покупку премиума:**  
# *(из первого визита в первую покупку премиума)*

# In[111]:


pd.DataFrame({'Группа': ['Контрольная 1', 'Контрольная 2', 'Тестовая'], 'Конверсия в первую покупку премиума, %': 
              [round(control_group_1_premium_newusers_3.query('country_x not in @country_list').product_type.count() / control_group_1.query('country_x not in @country_list').uid.nunique() * 100, 2),
               round(control_group_2_premium_newusers_3.query('country_x not in @country_list').product_type.count() / control_group_2.query('country_x not in @country_list').uid.nunique() * 100, 2), 
               round(test_group_premium_newusers_3.query('country_x not in @country_list').product_type.count() / test_group.query('country_x not in @country_list').uid.nunique() * 100, 2)]})


# ------------------
# **Средняя выручка на пользователя (ARPU):**

# In[112]:


# средняя выручка на пользователя по группам (total по всем продуктам):

rev_1 = df_merged.groupby(['group'], as_index = False).agg({'revenue': 'sum'})
rev_2 = df_merged.groupby(['group'], as_index = False).agg({'uid': 'nunique'})
rev_mean = rev_1.merge(rev_2)
rev_mean['ARPU'] = round(rev_mean.revenue / rev_mean.uid, 2)
rev_mean.drop(['revenue', 'uid'], axis= 1, inplace= True)

rev_mean


# In[113]:


# средняя выручка от премиума (premium_no_trial) на пользователя по группам:

rev_3 = df_merged.query('product_type == "premium_no_trial"').groupby(['group'], as_index = False).agg({'revenue': 'sum'})
rev_4 = df_merged.groupby(['group'], as_index = False).agg({'uid': 'nunique'})
rev_prem_mean = rev_3.merge(rev_4)
rev_prem_mean['ARPU_prem'] = round(rev_prem_mean.revenue / rev_prem_mean.uid, 2)
rev_prem_mean.drop(['revenue', 'uid'], axis= 1, inplace= True)

rev_prem_mean


# In[114]:


# Средняя выручка на платящего пользователя в разрезе по продуктам и группам (ARPPU)
# (привожу лишь для наглядности, тут можно отметить, что в целом по всем продуктам 
# средняя выручка с платяшего пользователя увеличилась (не только по премиуму).
    
rev_mean_by_product = df_merged.groupby(['product_type', 'group'], as_index = False).agg({'revenue': 'mean'}) .pivot_table('revenue', index = 'product_type', columns = 'group', aggfunc = 'mean') .round()

rev_mean_by_product


# -----------------
# **средняя кол-во покупок премиума на пользователя по группам:**

# In[115]:


# список пользователей, которые покупали премиум (не пробный):

bought_premium_uids_2 = df_merged.query('product_type == "premium_no_trial"').uid.tolist()


# Таблица с количеством покупок премиума на пользователя:

prem_purch_count = df_merged.query('uid in @bought_premium_uids_2 and product_type == "premium_no_trial"') .groupby(['uid', 'group'], as_index = False).agg({'product_type': 'count'}) .sort_values('product_type', ascending = False) .rename(columns = {'product_type': 'num_of_purchases'})


# среднее кол-во покупок премиума на пользователя в группах:

round(prem_purch_count.groupby('group', as_index = False).agg({'num_of_purchases': 'mean'}), 2)


# -----------------------
# # A/A-тесты

# Сначала проведем АА-анализ на двух контрольных группах.  
# Если они значимо не различаются по целевым метрикам - это будет значить, что тест проведен корректно по части сплитования.
# И тогда в дальнейшем сможем сравнивать тестовую группу с одной из контрольных, либо с объединенной контрольной (состоящей из двух контрольных) ((AB-тест).
# 
# -----------

# В объединенной таблице df_merged оставим только новых пользователей (т.е. только первые записи о каждом id),  
# а также исключим страны из списка выше и список подозрительных uid: 

# In[116]:


# Таблица с новыми пользователями (уникальные uid) с исключением всех "подозрительных" uid и стран.

df_merged = df_merged.sort_values('paid_at').drop_duplicates(subset=['uid']).query('country_x not in @country_list and uid not in @suspicious_uids')

df_merged.head()


# # A/A-тест: CR в первую покупку премиума

# In[117]:


# Посчитаем уников и покупки премиума, сгруппируем по группам эксперимента.
# Создадим столбец с конверсией, которая в рамках каждого uid будет принимать значения 0 или 1 (была покупка премиума или нет).

# Уники
a = df_merged.groupby(['group', 'uid'], as_index = False).agg({'age': 'nunique'})


# Покупки премиума
b = df_merged .sort_values('paid_at') .drop_duplicates(subset=['uid']) .query('product_type == "premium_no_trial"') .groupby(['group', 'uid'], as_index = False) .agg({'age': 'nunique'})

# Соединяем и добавляем конверсию:
stat = a.merge(b, on = ['group', 'uid'], how = 'left').rename(columns = {'age_x': 'first_visit', 'age_y': 'premium_purchases'})
stat["CR"] = stat.premium_purchases / stat.first_visit
stat = stat.fillna(0)

stat


# In[118]:


# Та же таблица только с обеими контрольными группами (без тестовой): 

stat_both_control = stat.query('group != "test"')


# In[119]:


# Таблица сопряженности, 
# показывающая количество уникальных юзеров с конверсией в покупку премиума и без - по двум контрольным группам.

pd.crosstab(stat_both_control.group, stat_both_control.CR, values=stat_both_control.CR, aggfunc='count')


# Проверим, можно ли сказать, что есть зависимость переменных (группа эксперимента и конверсия в покупку премиума).  
# => Сравниваем качественные метрики 
# и применяем самый подходящий на такой случай тест - критерий согласия Пирсона (хи-квадрат).
# 
# Нулевая гипотеза: переменные не связаны друг с другом.  
# Альтернативная гипотеза: есть связь между переменными.
# 
# За порог значимости (α) здесь и далее принимаем значение 0.05.

# In[120]:


statistics, p, dof, expected = chi2_contingency(pd.crosstab(stat_both_control.group, stat_both_control.CR, values=stat_both_control.CR, aggfunc='count'))

statistics, p


# In[121]:


prob = 0.95
alpha = 1.0 - prob
if p <= alpha:
    print('Отклоняем H0')
else:
    print('Не отклоняем H0')


# ### p-value = 96%, следовательно нельзя утверждать, что нулевая гипотеза неверна, что значит, что у нас нет достаточных доказательств, чтобы сказать, что существует связь между группой эксперимента (в контроле) и конверсией в покупку премиума.

# --------------------
# # A/A-тест: средняя выручка на пользователя (ARPU)

# In[122]:


# т.к. в df_merged мы оставили только первые упоминания о каждом пользователе, 
# создадим новый df без такой фильтрации:

kk = test_group.append(control_group_1, sort = False).append(control_group_2, sort = False)
kk = kk.query('country_x not in @country_list and uid not in @suspicious_uids')


# In[123]:


# Таблица с суммами выручки на каждого пользователя по группам эксперимента:

total_revenue = kk.groupby(['group', 'uid'], as_index = False).agg({'revenue': 'sum'})

total_revenue.head()


# In[124]:


# Средняя выручка на пользователя по группам 
# (еще раз посмотрим на нее и убедимся, что выше, перед А/А-тестами, посчитали корректно):

total_revenue.groupby('group', as_index = False).agg({'revenue': 'mean'}).round(2)


# In[125]:


# Размеры групп: 

total_revenue.groupby('group').size()


# Размеры групп похожи и достаточны для того, чтобы при выборе теста не ограничиваться условием нормальности распределения.

# In[126]:


# тест на гомогенность данных между контрольными группами:

stats.levene(total_revenue.query('group == "control_1"').revenue, total_revenue.query('group == "control_2"').revenue, center='mean')


# Но имеем негомогенные данные между контрольными группами.   
# Применим t-критерий Уэлча.
# 

# Нулевая гипотеза: средние значения выручки на пользователя в двух контрольных группах равны.  
# Альтернативная гипотеза: средние значения выручки на пользователя в двух контрольных группах не равны.

# In[127]:


stats.ttest_ind(total_revenue.query('group == "control_1"').revenue, total_revenue.query('group == "control_2"').revenue, equal_var=False)


# ### На основании значения pvalue (> 0.05) не отвергаем нулевую гипотезу и считаем, что изменение средней выручки на пользователя между контрольными группыми можно назвать не стат. значимым.

# ---------------------------
# # A/A-тест: средняя выручка от покупок премиума на пользователя (ARPU_prem)

# In[128]:


# для дальнейших расчетов создадим доп. колонку в таблице kk, показывающую выручку от премиума в каждой строке:

kk['prem_revenue']  = np.where(kk['product_type'] == 'premium_no_trial', kk.revenue, 0)

kk.head()


# In[129]:


# Таблица с суммами выручки с покупок премиума (premium_no_trial) на каждого пользователя по группам эксперимента:

premium_revenue = kk.groupby(['group', 'uid'], as_index = False).agg({'prem_revenue': 'sum'})

premium_revenue.head()


# In[130]:


# Размеры групп: 

premium_revenue.groupby('group').size()


# In[131]:


# тест на гомогенность данных между контрольными группами:

stats.levene(premium_revenue.query('group == "control_1"').prem_revenue, premium_revenue.query('group == "control_2"').prem_revenue, center='mean')


# Размеры групп похожи, достаточны для того, чтобы при выборе теста не ограничиваться условием нормальности распределения, данные гомогенны.  
# Применяем стандартный t-test.

# Нулевая гипотеза: средние значения выручки c премиума на пользователя в двух контрольных группах равны.  
# Альтернативная гипотеза: средние значения выручки c премиума на пользователя в двух контрольных группах не равны.

# In[132]:


stats.ttest_ind(premium_revenue.query('group == "control_1"').prem_revenue, premium_revenue.query('group == "control_2"').prem_revenue, equal_var=True)


# ### На основании значения pvalue (> 0.05) не отвергаем нулевую гипотезу и считаем, что изменение средней выручки c премиума на пользователя между контрольными группыми нельзя назвать стат. значимым.

# ------------------
# # A/A-тест: повторные покупки премиума 
# 
# 

# In[133]:


# Таблица с количеством покупок премиума на пользователя:

prem_purch_count


# In[134]:


# Вспомним среднее кол-во покупок премиума на пользователя в группах:

round(prem_purch_count.groupby('group', as_index = False).agg({'num_of_purchases': 'mean'}), 2)


# Видим, что в тестовой группе значение немного больше, чем в контрольных
# (отдельный вопрос, конечно, почему в целом значения такие низкие, при условии, что данные за год, а подписка ежемесячная,  
# но это уже вопрос к отдельному исследованию про ретеншн и отток пользователей).
# 
# Проверим стат. значимость различия метрики между контрольными группами:
# 

# In[135]:


# Размерность групп:

prem_purch_count.groupby('group').size()


# In[136]:


# тест на гомогенность данных между контрольными группами:

stats.levene(prem_purch_count.query('group == "control_1"').num_of_purchases, prem_purch_count.query('group == "control_2"').num_of_purchases, center='mean')


# Данных в группах достаточно, но они не гомогенные и не совсем похожи по размеру. Применим в этот раз bootstrap.
# 
# Нулевая гипотеза: среднее кол-во покупок премиум-тарифа в контрольных группах одинаковое.  
# Альтернативная гипотеза: среднее кол-во покупок премиум-тарифа в контрольных группах различается.
# 
# 

# In[137]:


# Функция для bootstrap

def get_bootstrap(
    data_column_1, # числовые значения первой выборки
    data_column_2, # числовые значения второй выборки
    boot_it = 10000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
):
    boot_len = max([len(data_column_1), len(data_column_2)])
    boot_data = []
    for i in tqdm(range(boot_it)): # извлекаем подвыборки
        samples_1 = data_column_1.sample(
            boot_len, 
            replace = True # параметр возвращения
        ).values
        
        samples_2 = data_column_2.sample(
            boot_len, # чтобы сохранить дисперсию, берем такой же размер выборки
            replace = True
        ).values
        
        boot_data.append(statistic(samples_1-samples_2)) 
    pd_boot_data = pd.DataFrame(boot_data)
        
    left_quant = (1 - bootstrap_conf_level)/2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])
        
    p_1 = norm.cdf(
        x = 0, 
        loc = np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_2 = norm.cdf(
        x = 0, 
        loc = -np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2
        
    # Визуализация
    _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
    for bar in bars:
        if abs(bar.get_x()) <= quants.iloc[0][0] or abs(bar.get_x()) >= quants.iloc[1][0]:
            bar.set_facecolor('red')
        else: 
            bar.set_facecolor('grey')
            bar.set_edgecolor('black')
    
    plt.style.use('ggplot')
    plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
    plt.xlabel('boot_data')
    plt.ylabel('frequency')
    plt.title("Histogram of boot_data")
    plt.show()
       
    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value}


# In[138]:


rep_purch_AA = get_bootstrap(prem_purch_count.query('group == "control_1"').num_of_purchases,
                                 prem_purch_count.query('group == "control_2"').num_of_purchases)


# In[139]:


rep_purch_AA_p = round(rep_purch_AA['p_value'],3)
rep_purch_c1 = round(prem_purch_count.query('group=="control_1"').num_of_purchases.mean(),2)
rep_purch_c2 = round(prem_purch_count.query('group=="control_2"').num_of_purchases.mean(),2)
print(f'Среднее кол-во повторных покупок для контрольной группы №1 равно {rep_purch_c1}, для контрольной группы №2 - {rep_purch_c2}')
print(f'p-value bootstrap-теста = {rep_purch_AA_p}')


# ### p-value > 5%, что не дает нам оснований утверждать, что нулевая гипотеза неверна.  
# ### Т.е. среднее кол-во покупок премиум-тарифа в двух контрольных группах значимо не различается.

# ----------------
# # A/A-тесты: резюме
# В результате проверки выше можно утверждать, что сплитование пользователей было проведено корректно,  
# и можем переходить к сравнению тестовой группы с контрольми. 
# 
# --------------

# # А/B-тесты

# # А/B-тест: CR в первую покупку премиума
# 

# In[140]:


# таблица с конверсиями в первую покупку премиума по группам и пользователям:
stat


# In[141]:


# Заменим значения control_1 и control_2 на "control", тем самым объединив две контрольные группы в одну:

stat_2 = stat.replace(['control_1', 'control_2'],' control')


# In[142]:


# Таблица сопряженности, 
# показывающая количество уникальных юзеров с конверсией в покупку премиума и без - суммарно в контрольных группах и в тестовой.

pd.crosstab(stat_2.group, stat_2.CR, values=stat_2.CR, aggfunc='count')


# Проверим, можно ли сказать, что есть зависимость переменных (группа эксперимента и конверсия в покупку премиума).
# => Сравниваем качественные метрики, 
# и снова применяем критерий согласия Пирсона (хи-квадрат).
# 
# Нулевая гипотеза: переменные не связаны друг с другом.  
# Альтернативная гипотеза: есть связь между переменными.

# In[143]:


statistics, p, dof, expected = chi2_contingency(pd.crosstab(stat_2.group, stat_2.CR, values=stat_2.CR, aggfunc='count'))

statistics, p


# In[144]:


prob = 0.95
alpha = 1.0 - prob
if p <= alpha:
    print('Отклоняем H0')
else:
    print('Не отклоняем H0')


# ### на основе значения pvalue (< 0.05) отвергаем нулевую гипотезу и считаем, что при том, что конверсия в тестовой группе упала относительно конверсии в контрольных группах, различия  являются статистически значимыми.
# 

# ----------------
# # A/B-тест: средняя выручка на пользователя (ARPU)

# In[145]:


# тест на гомогенность данных между контрольными и тестовой группами:

stats.levene(total_revenue.query('group != "test"').revenue, total_revenue.query('group == "test"').revenue, center='mean')


# Размеры групп похожи и достаточны для того, чтобы при выборе теста не ограничиваться условием нормальности распределения.
# Но имеем негомогенные данные между контрольными группами.  
# Снова применим t-критерий Уэлча.  
# 
# Нулевая гипотеза: средние значения выручки на пользователя в контрольных и тестовой группах равны.  
# Альтернативная гипотеза: средние значения выручки на пользователя в контрольных и тестовой группах не равны.

# In[146]:


stats.ttest_ind(total_revenue.query('group != "test"').revenue, total_revenue.query('group == "test"').revenue, equal_var=False)


# ### т.к. p-value > 0.05, не отвергаем нулевую гипотезу и считаем, что изменение (увеличение) средней выручки на пользователя в тестовой группе нельзя назвать стат. значимым.
# 

# -----------------------
# # A/B-тест: средняя выручка от покупок премиума на пользователя (ARPU_prem)

# In[147]:


# тест на гомогенность данных между контрольными группами:

stats.levene(premium_revenue.query('group == "control_1"').prem_revenue, premium_revenue.query('group == "control_2"').prem_revenue, center='mean')


# Размеры групп похожи, достаточны для того, чтобы при выборе теста не ограничиваться условием нормальности распределения (эти условия проверены выше при A/A-тесте), данные гомогенны. Применяем t-test.  
# 
# Нулевая гипотеза: средние значения выручки с премиума на пользователя в контрольных и тестовой группах равны.  
# Альтернативная гипотеза: средние значения выручки с премиума на пользователя в контрольных и тестовой группах не равны.

# In[148]:


stats.ttest_ind(premium_revenue.query('group != "test"').prem_revenue, premium_revenue.query('group == "test"').prem_revenue, equal_var=True)


# ### т.к. p-value > 0.05, не отвергаем нулевую гипотезу и считаем, что изменение (увеличение) средней выручки с премиума на пользователя в тестовой группе также можно назвать не стат. значимым.

# _______________
# # A/B-тест: повторные покупки премиума 
# 

# In[149]:


# Таблица с количеством покупок премиума на пользователя:

prem_purch_count


# In[150]:


# тест на гомогенность данных между тестовой и первой контрольной группами:

stats.levene(prem_purch_count.query('group == "test"').num_of_purchases, prem_purch_count.query('group == "control_1"').num_of_purchases, center='mean')


# In[151]:


# тест на гомогенность данных между тестовой и второй контрольной группами:

stats.levene(prem_purch_count.query('group == "test"').num_of_purchases, prem_purch_count.query('group == "control_2"').num_of_purchases, center='mean')


# In[152]:


# тест на гомогенность данных между тестовой и склеенной контрольной группами:

stats.levene(prem_purch_count.query('group == "test"').num_of_purchases, prem_purch_count.query('group != "test"').num_of_purchases, center='mean')


# Имеем достаточное кол-во данных, но гомогенность подтвердилась только в сравнении с первой контрольной группой. Можем сравнить средние тестовой и первой контрольной группы с применением t-test.  
# 
# 
# Нулевая гипотеза: среднее кол-во покупок премиум-тарифа в тестовой и контрольной группах одинаковое.  
# Альтернативная гипотеза: среднее кол-во покупок премиум-тарифа в тестовой и контрольной группах различается.

# In[153]:


stats.ttest_ind(prem_purch_count.query('group == "test"').num_of_purchases, prem_purch_count.query('group == "control_1"').num_of_purchases)


# А сравнение со второй контрольной группой и со склеенной группой (состоящей из двух контрольных) проведем с помощью t-критерия Уэлча:

# In[154]:


stats.ttest_ind(prem_purch_count.query('group == "test"').num_of_purchases, prem_purch_count.query('group == "control_2"').num_of_purchases, equal_var=False)


# In[155]:


stats.ttest_ind(prem_purch_count.query('group != "test"').num_of_purchases, prem_purch_count.query('group == "test"').num_of_purchases, equal_var=False)


# ### во всех случаях тут p-value > 5%, что не дает нам оснований утверждать, что нулевая гипотеза неверна. Т.е. среднее кол-во покупок премиум-тарифа в тестовой и контрольной группах значимо не различается.
# 

# --------------------------------------

# ----------------------------
# # Выводы:
# 
# **В результате эксперимента**  
#    
# * **конверсия в оплату премиума статистически значимо снизилась** (1.66% - 1.7% в контроле VS 1.1% в тесте), 
# * **средняя выручка на пользователя увеличилась НЕ статистически значимо** (347.32 и 261.4 в контроле VS 402.93 в тесте), 
# * **средняя выручка с премиума на пользователя увеличилась НЕ статистически значимо** (103 и 108.52 в контроле VS 146.11 в тесте),  
# * **проведение эксперимента не повлияло на кол-во повторных покупок премиума,**
# 
# **следовательно, эксперимент признаем НЕ успешным, изменения рекомендуется откатить и не масштабировать**,
# 
# т.к. при значимом снижении конверсии в долгосрочной перспективе даже увеличенная в моменте выручка (на выборке) покажет обратный эффект (в генеральной совокупности) и с большой вероятностью может негативно отразиться на показателях бизнеса.  
# 
# 
# ----------
# P.S.:  
# Несмотря на репрезентативность выборок и корректно проведенный тест, перед проведением новых экспериментов стоит дополнительно проанализировать (вооружившись более точными / полными данными):  
# Retention Rate, доли новых и вернувшихся пользователей.
# 
# Какие показатели стоит обогатить / пофиксить:  
# дата регистрации   
# тип платежной системы  
# was_premium   
# is_premium   
# 
