
from scipy.spatial import distance
import numpy as np
import cv2
from multiprocessing.pool import ThreadPool

import pandas as pd

def find_subsequent_numbers(numbers):
    from itertools import groupby
    from operator import itemgetter

    for k, g in groupby(enumerate(numbers), lambda x: x[0] - x[1]):

        print(list(map(lambda x:x[1],g)))


def is_vector_pointing_towards_point(vector_point1,vector_point2,point):

    distance_vector_point1_to_point = np.linalg.norm(np.array(vector_point1) - np.array(point))

    distance_vector_point2_to_point = np.linalg.norm(np.array(vector_point2) - np.array(point))

    return distance_vector_point2_to_point < distance_vector_point1_to_point



def thread_run_test():
    try:


        print("running thread")
        raise Exception("Hey Exception happened")
    except Exception as e:
        print(e)

def numpy_array_selection():
    arr = np.arange(20).reshape(4, 5)

    rids = [1,2,3]
    cids = [1,2,3]

    print(arr[rids,cids])


def loop_diag():
    arr = np.arange(25).reshape(-1, 5)

    for row in range(5):
        for col in range(row):
            arr[row,col] = 0

    print(arr)


def calculate_cosine_distance(u,v):

    u = np.array(u)
    v = np.array(v)


    dist = (1.0 - np.dot(u,v) / np.sqrt(np.dot(u,u) * np.dot(v,v))) / 2

    return dist

def check_parantheses(literal):

    bracket_stack = []

    bracket_dict = {"]": "[", ")": "(" , "}" : "{"}

    for c in literal:

        if c in set(["(", "[", "{"]):
            bracket_stack.append(c)

        else:

            if len(bracket_stack) == 0:
                return False
            else:
                last_inserted_bracked = bracket_stack.pop()


            if last_inserted_bracked != bracket_dict[c]:
                return False

    if len(bracket_stack) != 0:
        return False

    else:
        return True

def pandas_count_test():
    import pandas as pd
    df = pd.DataFrame({"frame_no_cam": [0, 0, 1, 1, 1],
                        "person_id": [0, 1, 0, 1, 2]}
                      )
    df = df.groupby(["frame_no_cam"],as_index=False).count()

    import matplotlib.pyplot as plt

    # x axis values
    x = list(df["frame_no_cam"])
    # corresponding y axis values
    y = list(df["person_id"])

    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')

    # giving a title to my graph
    plt.title('My first graph!')

    # function to show the plot
    plt.show()

    print(df)


class Solution:
    def regex_matching(self, s,p):

        len_p = len(p)
        len_s = len(s)

        dp = [[0] * (len_p + 1) for _ in range(len_s + 1)]

        dp[0][0] = 1

        for x in range(1, len_p + 1):
            if p[x - 1] == '*':
                dp[0][x] = dp[0][x - 1]

        for i in range(1, len_s + 1):
            for j in range(1, len_p + 1):
                if s[i - 1] == p[j - 1] or p[j - 1] == '?':
                    dp[i][j] = dp[i - 1][j - 1]

                elif p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 1] or dp[i - 1][j]

        return bool(dp[-1][-1])


def maxSubsetSum(arr):
    dp = []
    dp.append(arr[0])
    dp.append(max(arr[:2]))
    ans = max(dp)
    for a in arr[2:]:
        dp.append(max(max(dp[-2]+a, a), ans))
        ans = max(ans, dp[-1])
    return ans


def get_median(lst):
    lstLen = len(lst)
    index = (lstLen) // 2

    if (lstLen % 2) == 1:
        return lst[index]
    else:
        return (lst[index] + lst[index + 1]) / 2.0



def my_func(**kwargs):

    for key,val in kwargs.items():
        print("key {} val {}".format(key,val))

if __name__ == "__main__":
    import os
    print(os.path.basename(__file__).replace(".py",""))












