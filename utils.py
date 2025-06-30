# utils.py
import numpy as np
from numba import jit

# Numba JIT 컴파일러를 적용하여 계산 속도를 극대화합니다.
# 이 함수는 순수한 숫자 계산에 특화되어 있습니다.
@jit(nopython=True)
def calculate_similarity_jit(alleles1, alleles2):
    """ 두 유전자 배열 간의 유사도를 Jaccard 유사도 방식으로 빠르게 계산합니다. """
    # Numba는 collections.Counter를 지원하지 않으므로, 수동으로 구현
    intersection_size = 0
    
    # numpy를 활용하여 효율적으로 교집합 크기 계산
    unique_alleles1, counts1 = np.unique(alleles1, return_counts=True)
    unique_alleles2, counts2 = np.unique(alleles2, return_counts=True)
    
    for i in range(len(unique_alleles1)):
        for j in range(len(unique_alleles2)):
            if unique_alleles1[i] == unique_alleles2[j]:
                intersection_size += min(counts1[i], counts2[j])
                break

    union_size = len(alleles1) + len(alleles2) - intersection_size
    if union_size == 0:
        return 1.0
    return intersection_size / union_size
