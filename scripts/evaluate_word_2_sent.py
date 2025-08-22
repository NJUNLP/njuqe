import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, zscore
from transformers


def read_prob_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    probs = [list(map(float, line.split())) for line in lines]
    return probs

def read_z_score_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    z_scores = zscore([float(line.strip()) for line in lines])
    return z_scores

# 将 prob 转化为严重程度标签

def write_scores_file(file_path, scores):
    with open(file_path, 'w') as f:
        for score in scores:
            f.write(f"{score}\n")


def categorize_probs(probs, thresholds):
    result = []
    for prob_line in probs:
        labels = [0, ]
        for prob in prob_line:
            if prob < thresholds['critical'] and labels[-1] != 10:
                labels.append(10)
            elif prob < thresholds['major'] and labels[-1] != 5:
                labels.append(5)
            elif prob < thresholds['minor'] and labels[-1] != 1:
                labels.append(1)
        result.append(labels)
    return result

# 计算 MQM 分数


def calculate_mqm(results, word_nums):
    mqm_score = []
    for result, word_num in zip(results, word_nums):
        mqm_score.append(1 - sum(result) / word_num)

    return np.array(mqm_score)

# 找到使得 Spearman 相关系数最大的阈值


def find_best_thresholds(probs, ref_scores):
    best_thresholds = None
    best_scores= None
    best_spearman = -1

    cnt = 0
    for critical in np.arange(0.1, 1.00, 0.05):
        for major in np.arange(0.01 + critical, 1.00, 0.01):
            flag = False
            for minor in np.arange(0.01 + major, 1.00, 0.01):
                thresholds = {'critical': critical,
                              'major': major, 'minor': minor}
                results = categorize_probs(probs, thresholds)
                mqm_scores = calculate_mqm(results, [len(_) for _ in probs])

                # 计算 Spearman 相关系数
                spearman_corr, _ = spearmanr(ref_scores, mqm_scores)

                if spearman_corr > best_spearman:
                    flag = True
                    best_spearman = spearman_corr
                    best_thresholds = thresholds
                    best_scores = mqm_scores
                    cnt = 0
                    print(f"Spearman Correlation:{best_spearman:.4f}, Thresholds:{thresholds}")
                else:
                    cnt += 1
                    
                if cnt >= 20 and flag:
                    break

    return best_thresholds, best_spearman, best_scores

# 主函数


def find_best_combine(best_word_scores, best_regress_scores, ref_scores):
    best_spearman = -1
    best_scores = None
    for i in np.arange(0, 0.51, 0.1):
        mqm_scores = best_regress_scores * i + best_word_scores * (1 - i)
        # 计算 Spearman 相关系数
        spearman_corr, _ = spearmanr(ref_scores, mqm_scores)

        if spearman_corr > best_spearman:
            best_spearman = spearman_corr
            best_scores = mqm_scores
            print(f"Spearman Correlation:{best_spearman:.4f}, Thresholds:{i}")
    
    return best_scores


def main(prob_file_path, z_score_file_path, regress_file_path):
    probs = read_prob_file(prob_file_path)
    ref_scores = read_z_score_file(z_score_file_path)
    regress_scores = read_z_score_file(regress_file_path)

    # find the best threshold sets on valid set
    # best_thresholds, best_spearman_scores, best_scores = find_best_thresholds(
    #     probs, ref_scores)
    
    best_thresholds = {'critical': , 'major': , 'minor': }
    results = categorize_probs(probs, best_thresholds)
    best_word_scores = calculate_mqm(results, [len(_) for _ in probs])
    best_scores = find_best_combine(best_word_scores, regress_scores, ref_scores)
    
    with open("best.hter", "w") as f:
        for score in best_scores:
            f.write(f"{score}\n")
            
    # write_scores_file("best.mqm_score", best_scores)
    print(f"""Best Spearman Correlation:{spearmanr(ref_scores, best_scores)[0]:.4f}
Best Pearson Correlation:{pearsonr(ref_scores, best_scores)[0]:.4f}
Best Thresholds:{best_thresholds}""")


# 示例运行
# 请替换以下参数

prob_file_path = '.prob'
z_score_file_path = '.mqm_score'
regress_file_path = '.hter'

main(prob_file_path, z_score_file_path, regress_file_path)
