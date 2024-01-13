#程式參考老師java資源實作
def matrix_print(m):
    for row in m:
        print(row)

def str_set(s, i, c):
    return s[:i] + c + s[i+1:]

def find_path(m, x, y):
    print("=========================")
    print(f"x={x} y={y}")
    matrix_print(m)

    if x >= 6 or y >= 8:
        return False
    if m[x][y] == '*':
        return False
    if m[x][y] == '+':
        return False

    if m[x][y] == ' ':
        m[x] = str_set(m[x], y, '.')

    if m[x][y] == '.' and (x == 5 or y == 7):
        return True

    if y < 7 and m[x][y+1] == ' ':  # 向右
        if find_path(m, x, y+1):
            return True
    if x < 5 and m[x+1][y] == ' ':  # 向下
        if find_path(m, x+1, y):
            return True
    if y > 0 and m[x][y-1] == ' ':  # 向左
        if find_path(m, x, y-1):
            return True
    if x > 0 and m[x-1][y] == ' ':  # 向上
        if find_path(m, x-1, y):
            return True

    m[x] = str_set(m[x], y, '+')
    return False

maze = ["********",
        "** * ***",
        "     ***",
        "* ******",
        "*     **",
        "***** **"]

find_path(maze, 2, 0)
print("=========================")
matrix_print(maze)
#測試結果：成功走出迷宮
'''
=========================
x=2 y=0
********
** * ***
     ***
* ******
*     **
***** **
=========================
x=2 y=1
********
** * ***
.    ***
* ******
*     **
***** **
=========================
x=2 y=2
********
** * ***
..   ***
* ******
*     **
***** **
=========================
x=2 y=3
********
** * ***
...  ***
* ******
*     **
***** **
=========================
x=2 y=4
********
** * ***
.... ***
* ******
*     **
***** **
=========================
x=1 y=4
********
** * ***
.....***
* ******
*     **
***** **
=========================
x=1 y=2
********
** *+***
...++***
* ******
*     **
***** **
=========================
x=3 y=1
********
**+*+***
..+++***
* ******
*     **
***** **
=========================
x=4 y=1
********
**+*+***
..+++***
*.******
*     **
***** **
=========================
x=4 y=2
********
**+*+***
..+++***
*.******
*.    **
***** **
=========================
x=4 y=3
********
**+*+***
..+++***
*.******
*..   **
***** **
=========================
x=4 y=4
********
**+*+***
..+++***
*.******
*...  **
***** **
=========================
x=4 y=5
********
**+*+***
..+++***
*.******
*.... **
***** **
=========================
x=5 y=5
********
**+*+***
..+++***
*.******
*.....**
***** **
=========================
********
**+*+***
..+++***
*.******
*.....**
*****.**
'''

