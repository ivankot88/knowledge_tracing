import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from collections import Counter, defaultdict

tqdm.pandas()

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

DATA_DIR = 'data'
SOURCE_DIR = '/home/ivankot/data/'
POPULAR_PROBLEMS_PERCENT = 0.5
COUNT_OF_POPULAR_SOLVED_TASKS = 50

def get_submit(source):
    try:
        with open(f'{SOURCE_DIR}/{source}') as f:
            file = f.read()
            return file
    except: 
        return ''

def compute_popular_problems(list_of_lists, popular_percent):
    total_lists = len(list_of_lists)
    freq_dict = defaultdict(int)
    for lst in list_of_lists:
        for num in lst:
            freq_dict[num] += 1

    result = []
    for num, freq in freq_dict.items():
        if freq >= total_lists * popular_percent:
            result.append(num)
    return result


def main():
    data = pd.read_csv(os.path.join(DATA_DIR, 'python_base_table.csv'))
    print('Collecting source code...')
    

    # delete all consultations
    data['academic_year'] = data['name'].apply(lambda x: x.split('-')[-1])
    data = data[data.academic_year.isin(['17/18','18/19','19/20','20/21','21/22','22/23'])]

    # delete all users, which continue one course more than 1 academic year
    counter = Counter(data.drop_duplicates(['user_id','academic_year']).user_id.values)
    users_to_delete = [key for key, value in dict(counter).items() if value > 1]
    data = data[~data.user_id.isin(users_to_delete)]


    problem_df = pd.pivot_table(data, index='user_id', values='problem_id', aggfunc=lambda x: x.tolist())
    problem_df['problem_id'] = problem_df['problem_id'].apply(lambda x: set(x)).apply(lambda x: list(x))

    # popular problem - problem, that solve at least POPULAR_PROBLEMS_PERCENT of all students
    popular_problems = set(compute_popular_problems(problem_df['problem_id'].values, POPULAR_PROBLEMS_PERCENT))
    problem_df['popular_tasks'] = problem_df['problem_id'].apply(lambda x: len(popular_problems & set(x)))

    # correct students - students, that solve at least COUNT_OF_POPULAR_SOLVED_TASKS
    data = data[data.user_id.isin(problem_df[problem_df['popular_tasks'] > COUNT_OF_POPULAR_SOLVED_TASKS].index)]

    # read code files from SOURCE_DIR path
    data['code'] = data['source'].parallel_apply(get_submit)
    data = data[data['code'] != '']

    # save to pickle format
    data[['user_id', 'timestamp', 'code']].to_pickle(os.path.join('data', 'python_submits.pkl'))
    
    print('Collecting problem statements...')
    problems = pd.read_csv('../data/problems.csv')
    problems_ids = data[['problem_id']].drop_duplicates()
    pd.merge(problems_ids, 
             problems, 
             left_on='problem_id', 
             right_on='id')[['problem_id','statement']].to_pickle(os.path.join(DATA_DIR,'python_statements.pkl'))
    

if __name__ == '__main__':
    main()
