import re

from pythonds import Stack


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    return a / b


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


operations = {'+': add, '-': sub, '*': mul, '/': div}
weight = {'(': 3, '*': 2, '/': 2, '+': 1, '-': 1, None: 0}
lpri = {'(': 1, '+': 3, '-': 3, '*': 5, '/': 5, ')': 6}
rpri = {'(': 6, '+': 2, '-': 2, '*': 4, '/': 4, ')': 1}

# Define the stack of data and the stack of operations

#数据栈与运算符栈
data_stack = []
operator_stack = []

postexp_stack=[]
st_stack=[]


def deal_data():
    op = operator_stack.pop()
    num2 = float(data_stack.pop())
    num1 = float(data_stack.pop())
    result = operations[op](num1, num2)
    data_stack.append(result)
    print("result=",result)
    return result
# TODO : 查找为什么有括号不能运算的Bug
def calculate(equation):
    global data_stack
    global operator_stack
    try:
        while equation:
            '''
            匹配数字开头+任意运算符+数字
            以括号开头+负数+任意运算符+数字
            '''
            cur = re.search(r"((^\-\.?\d*)|(^\d+\.?\d*)|(^\(\-\d+\.?\d*)|\(|\)|\+|\-|\*|/)", equation).group()
            print("postexp_stack=" ,postexp_stack)
            print("operator_stack=",operator_stack)

            if "(-" in cur:
                print("cur=", cur)

                bracket = cur[0]
                if  lpri[operator_stack[-1]] < rpri[bracket] :
                    #括号入栈
                    operator_stack.append(bracket)
                elif lpri[operator_stack[-1]] > rpri[bracket] :
                    postexp_stack.append(operator_stack.pop())
                #字符串减一
                equation = equation[1:]
                num = cur[1:]
                #数字入栈 ： 负数
                postexp_stack.append(num)#修改

                data_stack.append(num)
                #字符串减一
                equation = equation[len(num):]

            else: #一般情况
                print("cur=", cur)
                lenth = len(cur)
                if is_number(cur):  #为数字
                    postexp_stack.append(cur)  # 修改
                    data_stack.append(cur)
                    equation = equation[lenth:]
                elif cur == ")":  #为括号
                    if operator_stack[-1] == "(":  # 相等
                        print( "operator_stack[-1] == ( -->",operator_stack.pop())
                        equation = equation[lenth:]
                    elif  lpri[operator_stack[-1]] > rpri[cur]  :  #左大于右
                        postexp_stack.append(operator_stack.pop()) #退栈存放到postexp中
                    else :
                        operator_stack.append(cur)
                        equation = equation[lenth:]

                else: #为运算符合
                    if not (operator_stack):#operator_stack为空情况下, 直接将符号存入就好
                        operator_stack.append(cur)
                        equation = equation[lenth:]
                    else:#operator_stack 不为空

                        if lpri[operator_stack[-1]] > rpri[cur] :#比较权重
                            postexp_stack.append(operator_stack.pop())  # 退栈存放到postexp中

                        elif lpri[operator_stack[-1]] < rpri[cur] :
                            operator_stack.append(cur)
                            equation = equation[lenth:]

        while operator_stack:
            postexp_stack.append(operator_stack.pop())
        return postexp_stack
        # return result
    except (KeyError,IndexError):
        data_stack = []
        operator_stack = []
        return '?'



def compvalue(postexp_stack) :

    for pos in postexp_stack :
        print(st_stack)
        if pos == '+' :
            if len(st_stack) < 2 :
                return None
            a = st_stack.pop()
            b = st_stack.pop()
            c = a + b
            st_stack.append(c)
        elif pos == '-':
            if len(st_stack) < 2 :
                return None
            a = st_stack.pop()
            b = st_stack.pop()
            c = a - b
            st_stack.append(c)
        elif pos == '*':
            if len(st_stack) < 2 :
                return None
            a = st_stack.pop()
            b = st_stack.pop()
            c = a * b
            st_stack.append(c)
        elif pos == '/':
            if len(st_stack) < 2 :
                return None
            a = st_stack.pop()
            b = st_stack.pop()
            c = b / a
            st_stack.append(c)
        else :
            st_stack.append(float(pos))
    return st_stack


