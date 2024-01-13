#程式有經由ChatGPT輔助，經由網路資源https://rust-algo.club/levenshtein_distance/理解編輯距離
# 遞迴
def recursive_edit_distance(str1, str2, m, n):
    if m == 0:
        return n
    if n == 0:
        return m
    if str1[m-1] == str2[n-1]:
        return recursive_edit_distance(str1, str2, m-1, n-1) 
    return 1 + min(
        recursive_edit_distance(str1, str2, m, n-1),  
        recursive_edit_distance(str1, str2, m-1, n),  
        recursive_edit_distance(str1, str2, m-1, n-1)  
    )
# 動態規劃
def dp_edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])
    return dp[m][n]

# 測試
str1 = "kitten"
str2 = "sitting"

# 遞迴法
recursive_result = recursive_edit_distance(str1, str2, len(str1), len(str2))
print("遞迴法最小編輯距離:", recursive_result)

# 動態規劃法
dp_result = dp_edit_distance(str1, str2)
print("動態規劃法最小編輯距離:", dp_result)

#測試結果
'''
遞迴法最小編輯距離: 3
動態規劃法最小編輯距離: 3
'''
