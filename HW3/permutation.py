def p(e, current_p=[]):
    # 如果所有元素都在目前排列中，則列印排列
    if not e:
        print(current_p)
        return

    # 對每個剩餘的元素進行遞迴
    for i in range(len(e)):
        # 選擇一個元素
        current_e = e[i]

        # 移除已選擇的元素
        remaining_e = e[:i] + e[i + 1:]

        # 遞迴呼叫
        p(remaining_e, current_p + [current_e])

# 測試
elements_to_permute = [1, 2, 3]
p(elements_to_permute)
#測試結果：
'''
[1, 2, 3]
[1, 3, 2]
[2, 1, 3]
[2, 3, 1]
[3, 1, 2]
[3, 2, 1]
'''
